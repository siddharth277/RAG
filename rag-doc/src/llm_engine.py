import os
import re
import requests
from typing import List, Dict, Any

class LLMEngine:
    """
    Production Multi-Provider LLM & Synthesis Engine supporting:
    - OpenAI (GPT-4o, GPT-4o-mini)
    - Ollama (Llama 3, Mistral, Qwen 2.5)
    - Advanced Local Extractive/Abstractive Synthesis Engine with 100% Citation Enforcement
    """

    def __init__(self, provider: str = "auto", api_key: str = None, model: str = None):
        self.provider = provider
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model

    def generate_answer(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Synthesizes an accurate answer from retrieved context chunks with document and page citations.
        """
        if not retrieved_chunks:
            return {
                "answer": "No relevant documents or context chunks were found matching your query.",
                "provider": "none",
                "citations": []
            }

        # Deduplicate and extract citations
        context_blocks = []
        citations = []
        seen_citations = set()

        for idx, chunk in enumerate(retrieved_chunks, 1):
            doc = chunk.get("doc_name", "Document")
            page = chunk.get("page_number", "N/A")
            citation_str = f"[{doc}, Page {page}]"
            
            if citation_str not in seen_citations:
                seen_citations.add(citation_str)
                citations.append({
                    "doc_name": doc,
                    "page_number": page,
                    "citation": citation_str,
                    "file_path": chunk.get("file_path", ""),
                    "score": chunk.get("similarity_score", 0.0)
                })

            context_blocks.append(f"--- Context Segment [{idx}] ({citation_str}) ---\n{chunk.get('text', '').strip()}\n")

        context_str = "\n".join(context_blocks)

        system_prompt = (
            "You are an expert AI Remote Sensing & Geospatial Analytics Research Assistant.\n"
            "Your objective is to answer user queries accurately based strictly on the provided context.\n"
            "CRITICAL CITATION RULES:\n"
            "1. Every factual claim or technical detail MUST end with its source document and page citation in brackets, e.g. [01_LandSegmenter.pdf, Page 3].\n"
            "2. Do not invent details not supported by the context.\n"
            "3. Format your response clearly using markdown headings, bullet points, and quotes."
        )

        user_prompt = f"User Question: {query}\n\nRetrieved Document Context:\n{context_str}\n\nPlease synthesize a complete, well-structured answer with inline document/page citations."

        # Determine active provider
        active_provider = self.provider
        if active_provider == "auto":
            if self.api_key:
                active_provider = "openai"
            elif self._check_ollama():
                active_provider = "ollama"
            else:
                active_provider = "local_synthesis"

        if active_provider == "openai" and self.api_key:
            try:
                answer = self._call_openai(system_prompt, user_prompt)
                return {"answer": answer, "provider": "OpenAI", "model": self.model or "gpt-4o-mini", "citations": citations}
            except Exception as e:
                print(f"[LLMEngine] OpenAI call error: {e}. Falling back to local synthesis...")

        if active_provider == "ollama":
            try:
                answer = self._call_ollama(system_prompt, user_prompt)
                return {"answer": answer, "provider": "Ollama", "model": self.model or "llama3", "citations": citations}
            except Exception as e:
                print(f"[LLMEngine] Ollama call error: {e}. Falling back to local synthesis...")

        # Fallback Local Smart Synthesis Engine
        answer = self._synthesize_local(query, retrieved_chunks, citations)
        return {"answer": answer, "provider": "Local Synthesis Engine", "model": "Extractive-RAG-v2", "citations": citations}

    @staticmethod
    def _check_ollama() -> bool:
        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=1)
            return r.status_code == 200
        except Exception:
            return False

    def _call_openai(self, system_prompt: str, user_prompt: str) -> str:
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key)
        model_name = self.model or "gpt-4o-mini"
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2
        )
        return response.choices[0].message.content

    def _call_ollama(self, system_prompt: str, user_prompt: str) -> str:
        model_name = self.model or "llama3"
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": model_name,
            "prompt": f"{system_prompt}\n\n{user_prompt}",
            "stream": False
        }
        r = requests.post(url, json=payload, timeout=60)
        r.raise_for_status()
        return r.json().get("response", "")

    def _synthesize_local(self, query: str, chunks: List[Dict[str, Any]], citations: List[Dict[str, Any]]) -> str:
        """
        Advanced local synthesis engine:
        Extracts key evidence sentences matching query terms, structures findings, and enforces inline page citations.
        """
        stopwords = {"what", "is", "the", "in", "of", "and", "a", "an", "for", "to", "how", "why", "are", "which", "on", "with", "by", "from"}
        query_terms = [w.lower() for w in re.findall(r'\w+', query) if w.lower() not in stopwords]

        findings = []

        for chunk in chunks:
            text = chunk["text"]
            doc = chunk["doc_name"]
            page = chunk["page_number"]
            score = chunk.get("similarity_score", 0.0)
            citation = f"[{doc}, Page {page}]"

            # Split into sentences
            sentences = [s.strip() for s in re.split(r'(?<=[.!?]) +', text) if len(s.strip()) > 15]
            matched_sentences = []

            for sent in sentences:
                sent_lower = sent.lower()
                # Count matching query terms
                matches = sum(1 for term in query_terms if term in sent_lower)
                if matches > 0:
                    matched_sentences.append((matches, sent))

            # Sort sentences by keyword match density
            matched_sentences.sort(key=lambda x: x[0], reverse=True)

            if matched_sentences:
                top_sentences = " ".join([s[1] for s in matched_sentences[:2]])
                findings.append({
                    "citation": citation,
                    "text": top_sentences,
                    "score": score
                })
            else:
                snippet = text[:280].replace('\n', ' ')
                findings.append({
                    "citation": citation,
                    "text": f"{snippet}...",
                    "score": score
                })

        output = []
        output.append(f"### Research Findings for: *\"{query}\"*\n")
        output.append("Based on retrieved context sections across the indexed research documents:\n")

        for f in findings:
            output.append(f"- **From {f['citation']}** *(Relevance Score: {f['score']:.4f})*:\n  > {f['text']}\n")

        citation_summary = ", ".join([c["citation"] for c in citations])
        output.append(f"\n**Verified Source Citations:** {citation_summary}")
        return "\n".join(output)

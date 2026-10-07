import argparse
import sys
import os
import json

# Ensure UTF-8 stdout encoding for Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from src.rag_pipeline import RAGPipeline

def main():
    parser = argparse.ArgumentParser(
        description="PDF RAG Pipeline CLI (Improved Edition): Hybrid Search, Deduplication & Page Citations."
    )
    parser.add_argument("--ingest", action="store_true", help="Ingest all PDFs in data/documents with SHA256 deduplication and build FAISS+BM25 indices.")
    parser.add_argument("--query", "-q", type=str, help="User query to search against vector database.")
    parser.add_argument("--top-k", "-k", type=int, default=5, help="Number of top relevant chunks to retrieve (default: 5).")
    parser.add_argument("--no-hybrid", action="store_true", help="Disable BM25 hybrid search and use dense vector search only.")
    parser.add_argument("--info", action="store_true", help="Display vector database statistics.")
    parser.add_argument("--provider", type=str, default="auto", choices=["auto", "openai", "ollama", "local_synthesis"], help="LLM provider backend.")
    parser.add_argument("--api-key", type=str, default=None, help="OpenAI API Key (optional).")

    args = parser.parse_args()

    use_hybrid = not args.no_hybrid
    rag = RAGPipeline(llm_provider=args.provider, api_key=args.api_key, use_hybrid_search=use_hybrid)

    if args.ingest:
        print("[CLI] Starting document ingestion with SHA256 deduplication & hybrid indexing...")
        stats = rag.ingest_documents(force_reindex=True)
        print("\n[CLI] Ingestion completed:")
        print(json.dumps(stats, indent=2))
        return

    if args.info:
        stats = rag.get_stats()
        print("\n[CLI] Vector Database Statistics:")
        print(json.dumps(stats, indent=2))
        return

    if args.query:
        print(f"\n[CLI] Querying: '{args.query}' (Top-K={args.top_k}, Hybrid Search={use_hybrid}, LLM={args.provider})")
        res = rag.query(args.query, top_k=args.top_k)

        print("\n" + "="*75)
        print(f"QUESTION: {res['user_query']}")
        print(f"LLM BACKEND: {res['provider']} ({res.get('model', 'N/A')}) | HYBRID SEARCH: {res.get('hybrid_search_enabled', True)}")
        print("="*75)
        print("\nANSWER:\n")
        print(res['answer'])
        print("\n" + "-"*75)
        print("VERIFIED DOCUMENT & PAGE CITATIONS:")
        for c in res['citations']:
            print(f" - {c['citation']} (Similarity Score: {round(c['score'], 4)})")
        print("="*75)
        return

    if len(sys.argv) == 1:
        parser.print_help()

if __name__ == "__main__":
    main()

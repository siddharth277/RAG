import os
from typing import List, Dict, Any, Optional

from src.config import (
    DOCUMENTS_DIR, VECTOR_DB_DIR, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP,
    DEFAULT_EMBEDDING_MODEL, DEFAULT_TOP_K, ENABLE_DEDUPLICATION,
    ENABLE_HYBRID_SEARCH, ENABLE_MMR, MMR_LAMBDA, RRF_K_PARAM
)
from src.pdf_extractor import PDFExtractor
from src.text_chunker import TextChunker
from src.embeddings import EmbeddingManager
from src.vector_store import VectorStore
from src.llm_engine import LLMEngine

class RAGPipeline:
    """
    Improved Production-Grade End-to-End RAG Pipeline:
    PDFs -> Text Extraction (Deduplicated & Cleaned) -> Semantic Chunking -> Embeddings -> FAISS + BM25 Vector DB -> Hybrid Retrieval + MMR -> LLM Answer + Citations
    """

    def __init__(
        self,
        documents_dir: str = DOCUMENTS_DIR,
        vector_db_dir: str = VECTOR_DB_DIR,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        llm_provider: str = "auto",
        api_key: Optional[str] = None,
        use_hybrid_search: bool = ENABLE_HYBRID_SEARCH,
        use_mmr: bool = ENABLE_MMR
    ):
        self.documents_dir = documents_dir
        self.vector_db_dir = vector_db_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_hybrid_search = use_hybrid_search
        self.use_mmr = use_mmr
        
        self.extractor = PDFExtractor(documents_dir=documents_dir, enable_dedup=ENABLE_DEDUPLICATION)
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedder = EmbeddingManager(model_name=embedding_model)
        self.vector_store = VectorStore(vector_db_dir=vector_db_dir)
        self.llm_engine = LLMEngine(provider=llm_provider, api_key=api_key)

        # Attempt loading pre-built vector index from disk
        self.loaded = self.vector_store.load()

    def ingest_documents(self, force_reindex: bool = False) -> Dict[str, Any]:
        """
        Ingestion Workflow:
        1. Extract text page-by-page (De-hyphenated & Deduplicated)
        2. Chunk text semantically into overlapping blocks
        3. Generate dense embeddings
        4. Build FAISS & BM25 indices and save to disk
        """
        if self.loaded and not force_reindex:
            print("[RAGPipeline] Pre-built vector index loaded. Skipping re-indexing.")
            return self.vector_store.get_stats()

        print("[RAGPipeline] === Starting Document Ingestion & Indexing ===")
        # Step 1: Text Extraction with SHA256 Deduplication & Cleaning
        pages_data = self.extractor.extract_all(directory=self.documents_dir)
        if not pages_data:
            return {"error": "No valid pages extracted from PDFs."}

        # Step 2: Semantic Paragraph-Aware Chunking
        chunks = self.chunker.chunk_all_pages(pages_data)
        if not chunks:
            return {"error": "No text chunks generated."}

        # Step 3: Embeddings Generation
        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.embed_texts(texts)

        # Step 4: Hybrid Indexing & Storage
        self.vector_store.build_and_save(embeddings, chunks)
        self.loaded = True

        stats = self.vector_store.get_stats()
        print("[RAGPipeline] === Ingestion & Indexing Successfully Completed ===")
        return stats

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]:
        """
        Hybrid Top-K Retrieval:
        Combines FAISS Dense Embedding Search + BM25 Lexical Keyword Search with RRF and MMR.
        """
        if not self.loaded:
            self.ingest_documents()

        query_vector = self.embedder.embed_query(query)

        if self.use_hybrid_search:
            results = self.vector_store.hybrid_search(
                query_text=query,
                query_embedding=query_vector,
                top_k=top_k,
                rrf_k=RRF_K_PARAM,
                use_mmr=self.use_mmr,
                mmr_lambda=MMR_LAMBDA
            )
        else:
            results = self.vector_store.search_dense(query_vector, top_k=top_k)

        return results

    def query(self, user_query: str, top_k: int = DEFAULT_TOP_K) -> Dict[str, Any]:
        """
        End-to-End Query Execution:
        Query -> Hybrid Retrieval -> Multi-LLM Synthesis -> Answer + Document & Page Citations
        """
        if not self.loaded:
            self.ingest_documents()

        print(f"[RAGPipeline] Query: '{user_query}' | Top-K={top_k} | Hybrid Search={self.use_hybrid_search}")
        retrieved_chunks = self.retrieve(user_query, top_k=top_k)

        result = self.llm_engine.generate_answer(user_query, retrieved_chunks)
        result["user_query"] = user_query
        result["retrieved_chunks"] = retrieved_chunks
        result["top_k"] = top_k
        result["hybrid_search_enabled"] = self.use_hybrid_search
        return result

    def get_stats(self) -> Dict[str, Any]:
        return self.vector_store.get_stats()

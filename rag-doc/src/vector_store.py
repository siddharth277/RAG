import os
import json
import re
import numpy as np
import faiss
from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi

class VectorStore:
    """
    Production Hybrid Vector Database:
    - Dense Vector Search (FAISS IndexFlatIP)
    - Sparse Lexical Search (BM25Okapi)
    - Reciprocal Rank Fusion (RRF)
    - Maximal Marginal Relevance (MMR) Diversification
    """

    def __init__(self, vector_db_dir: str = "vector_db"):
        self.vector_db_dir = vector_db_dir
        self.index_path = os.path.join(vector_db_dir, "index.faiss")
        self.metadata_path = os.path.join(vector_db_dir, "metadata.json")
        self.embeddings_path = os.path.join(vector_db_dir, "embeddings.npy")
        
        self.index = None
        self.metadata: List[Dict[str, Any]] = []
        self.embeddings: np.ndarray = None
        self.bm25 = None

    def tokenize(self, text: str) -> List[str]:
        """Simple lowercase word tokenizer for BM25."""
        return re.findall(r'\w+', text.lower())

    def build_and_save(self, embeddings: np.ndarray, chunks_metadata: List[Dict[str, Any]]):
        """
        Build FAISS dense vector index and BM25 index, then persist to disk.
        """
        if len(embeddings) == 0:
            print("[VectorStore] Warning: Empty embeddings array provided.")
            return

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        self.metadata = chunks_metadata
        self.embeddings = embeddings

        # Initialize BM25 corpus index
        corpus = [self.tokenize(c["text"]) for c in chunks_metadata]
        self.bm25 = BM25Okapi(corpus)

        os.makedirs(self.vector_db_dir, exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        np.save(self.embeddings_path, self.embeddings)
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

        print(f"[VectorStore] Indexing complete: {self.index.ntotal} vectors stored in FAISS & BM25 indices.")

    def load(self) -> bool:
        """
        Load FAISS index, raw embeddings array, and metadata payload from disk.
        Builds BM25 index in memory.
        """
        if not (os.path.exists(self.index_path) and os.path.exists(self.metadata_path)):
            return False

        try:
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            
            if os.path.exists(self.embeddings_path):
                self.embeddings = np.load(self.embeddings_path)

            # Build BM25 index
            corpus = [self.tokenize(c["text"]) for c in self.metadata]
            self.bm25 = BM25Okapi(corpus)

            print(f"[VectorStore] Successfully loaded {self.index.ntotal} vectors from '{self.vector_db_dir}'.")
            return True
        except Exception as e:
            print(f"[VectorStore] Failed to load vector store from '{self.vector_db_dir}': {e}")
            return False

    def is_empty(self) -> bool:
        return self.index is None or self.index.ntotal == 0

    def _get_dense_rankings(self, query_embedding: np.ndarray, top_k: int = 20) -> List[Tuple[int, float]]:
        """Internal helper for raw FAISS similarity search returning (idx, score) tuples."""
        if self.is_empty():
            return []
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, k)
        return [(int(idx), float(score)) for score, idx in zip(scores[0], indices[0]) if idx >= 0]

    def search_dense(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Dense similarity search using FAISS.
        Returns List of chunk metadata dicts with similarity_score attached.
        """
        rankings = self._get_dense_rankings(query_embedding, top_k=top_k)
        results = []
        for idx, score in rankings:
            if idx < 0 or idx >= len(self.metadata):
                continue
            chunk_data = dict(self.metadata[idx])
            chunk_data["similarity_score"] = float(score)
            results.append(chunk_data)
        return results

    def search_sparse(self, query_text: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """Sparse BM25 search returning (idx, bm25_score) tuples."""
        if not self.bm25:
            return []
        tokens = self.tokenize(query_text)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]

    def hybrid_search(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        top_k: int = 5,
        rrf_k: int = 60,
        use_mmr: bool = True,
        mmr_lambda: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Hybrid Retrieval combining FAISS Dense Vector Search and BM25 Sparse Search via Reciprocal Rank Fusion (RRF).
        Applies Maximal Marginal Relevance (MMR) for diversity if enabled.
        """
        if self.is_empty():
            return []

        fetch_k = min(top_k * 4, self.index.ntotal)

        # 1. Retrieve Dense candidate rankings
        dense_hits = self._get_dense_rankings(query_embedding, top_k=fetch_k)
        
        # 2. Retrieve Sparse BM25 candidate rankings
        sparse_hits = self.search_sparse(query_text, top_k=fetch_k)

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores = {}

        for rank, (idx, _) in enumerate(dense_hits):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (rrf_k + rank + 1))

        for rank, (idx, _) in enumerate(sparse_hits):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (rrf_k + rank + 1))

        # Sort candidate indices by combined RRF score
        sorted_candidates = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        if not sorted_candidates:
            return []

        # 4. Maximal Marginal Relevance (MMR) for Result Diversity
        candidate_indices = [idx for idx, _ in sorted_candidates]

        if use_mmr and self.embeddings is not None and len(candidate_indices) > top_k:
            q_vec = query_embedding[0] # (dim,)
            selected_indices = []
            unselected = list(candidate_indices)

            while len(selected_indices) < top_k and unselected:
                best_score = -float("inf")
                best_idx = None

                for cand in unselected:
                    cand_vec = self.embeddings[cand]
                    # Cosine similarity to query
                    sim_query = float(np.dot(q_vec, cand_vec))
                    
                    # Max similarity to already selected items
                    if selected_indices:
                        sim_selected = max([float(np.dot(cand_vec, self.embeddings[sel])) for sel in selected_indices])
                    else:
                        sim_selected = 0.0

                    mmr_score = mmr_lambda * sim_query - (1 - mmr_lambda) * sim_selected

                    if mmr_score > best_score:
                        best_score = mmr_score
                        best_idx = cand

                if best_idx is not None:
                    selected_indices.append(best_idx)
                    unselected.remove(best_idx)
                else:
                    break

            final_indices = selected_indices
        else:
            final_indices = candidate_indices[:top_k]

        # 5. Format results
        results = []
        for idx in final_indices:
            if idx < 0 or idx >= len(self.metadata):
                continue
            chunk_data = dict(self.metadata[idx])
            # Dense score as primary similarity score metric
            dense_score = float(np.dot(query_embedding[0], self.embeddings[idx])) if self.embeddings is not None else 0.0
            chunk_data["similarity_score"] = dense_score
            chunk_data["rrf_score"] = float(rrf_scores.get(idx, 0.0))
            results.append(chunk_data)

        return results

    def get_stats(self) -> Dict[str, Any]:
        if self.is_empty():
            return {"total_chunks": 0, "total_documents": 0, "dimension": 0}

        docs = set(chunk["doc_name"] for chunk in self.metadata if "doc_name" in chunk)
        return {
            "total_chunks": self.index.ntotal,
            "total_documents": len(docs),
            "dimension": self.index.d,
            "documents": sorted(list(docs))
        }

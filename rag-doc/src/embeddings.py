import numpy as np
from typing import List, Union

class EmbeddingManager:
    """
    Manages vector embedding generation using sentence-transformers models.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        print(f"[EmbeddingManager] Loading embedding model '{self.model_name}'...")
        try:
            from sentence_transformers import SentenceTransformer
            try:
                self.model = SentenceTransformer(self.model_name)
            except Exception as net_err:
                print(f"[EmbeddingManager] Online model load failed ({net_err}). Retrying with local cache...")
                self.model = SentenceTransformer(self.model_name, local_files_only=True)

            print(f"[EmbeddingManager] Embedding model loaded successfully. Output dim: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            raise RuntimeError(f"Failed to load sentence transformer model '{self.model_name}': {e}")

    def embed_texts(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """
        Generate L2-normalized vector embeddings for a list of text strings.
        Returns a float32 numpy array of shape (N, dim).
        """
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate L2-normalized vector embedding for a single query string.
        Returns a float32 numpy array of shape (1, dim).
        """
        embedding = self.model.encode(
            [query],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embedding.astype(np.float32)

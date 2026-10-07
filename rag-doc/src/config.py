# RAG Pipeline Configuration - Improved Production Edition
DOCUMENTS_DIR = "data/documents"
VECTOR_DB_DIR = "vector_db"

# Chunking parameters
DEFAULT_CHUNK_SIZE = 700
DEFAULT_CHUNK_OVERLAP = 120

# Embedding model defaults
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Retrieval & Hybrid Search Defaults
DEFAULT_TOP_K = 5
ENABLE_DEDUPLICATION = True
ENABLE_HYBRID_SEARCH = True  # BM25 + FAISS Dense Vector Search via Reciprocal Rank Fusion
RRF_K_PARAM = 60              # Reciprocal Rank Fusion constant
ENABLE_MMR = True             # Maximal Marginal Relevance for retrieval diversity
MMR_LAMBDA = 0.7              # Trade-off parameter between relevance (1.0) and diversity (0.0)

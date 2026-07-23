# Retrieval-Augmented Generation (RAG) PDF Pipeline

RAG (Retrieval-Augmented Generation) is an AI technique that improves the accuracy of large language models (LLMs) by fetching facts from an external knowledge base before generating a response.

In this repository, we created a RAG pipeline for reading a PDF and answering queries based on its contents.



## Major 4 steps

As we know , Building a RAG vector database involves 4 major steps:

1. Data Loading: Extracting text content from external documents (PDFs, text files, etc.).
2. Chunking: Breaking long documents into smaller, manageable text chunks.
3. Vector Embedding: Converting text chunks into dense mathematical vector representations.
4. Vector DB Storage: Indexing and storing embeddings in a database for fast similarity searches.



## Implementation

In this project, we used the following tools and libraries:

* Data Loading: PyPDFLoader (langchain_community) to extract pages from the PDF document.
* Chunking: RecursiveCharacterTextSplitter (langchain_text_splitters) with a chunk size of 1000 and chunk overlap of 200.
* Vector Embedding: HuggingFaceEmbeddings (langchain_huggingface) using the sentence-transformers/all-mpnet-base-v2 model.
* Vector DB:
  * InMemoryVectorStore (langchain_community) for volatile, fast in-memory indexing.
  * Chroma (langchain_chroma) with persistent local storage (./vector_db).



Author - SIDDHARTH SHUKLA 

github.com/siddharth277



import os
import streamlit as st
import pandas as pd
import json
import time

from src.rag_pipeline import RAGPipeline
from src.config import DOCUMENTS_DIR, VECTOR_DB_DIR

st.set_page_config(
    page_title="PDF RAG Pipeline Dashboard (Production Edition)",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for premium look & feel
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 24px;
        border-radius: 12px;
        color: white;
        margin-bottom: 24px;
        border: 1px solid #334155;
    }
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-top: 6px;
    }
    .metric-card {
        background-color: #1e293b;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #334155;
        text-align: center;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .citation-badge {
        background-color: #0369a1;
        color: #f0f9ff;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    .score-badge {
        background-color: #065f46;
        color: #ecfdf5;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .rrf-badge {
        background-color: #6d28d9;
        color: #f5f3ff;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .chunk-box {
        background-color: #0f172a;
        border-left: 4px solid #38bdf8;
        padding: 14px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_pipeline(chunk_size, chunk_overlap, embedding_model, use_hybrid, use_mmr):
    return RAGPipeline(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model=embedding_model,
        use_hybrid_search=use_hybrid,
        use_mmr=use_mmr
    )

def main():
    # Sidebar
    st.sidebar.title("⚙️ RAG Configuration")
    
    st.sidebar.subheader("LLM Settings")
    llm_provider = st.sidebar.selectbox(
        "LLM Provider",
        options=["auto", "local_synthesis", "openai", "ollama"],
        help="Select LLM backend for answer generation."
    )
    
    api_key = None
    if llm_provider in ["openai", "auto"]:
        api_key_input = st.sidebar.text_input("OpenAI API Key (optional)", type="password")
        if api_key_input:
            api_key = api_key_input

    st.sidebar.subheader("Retrieval & Search Features")
    use_hybrid = st.sidebar.toggle("⚡ Hybrid Search (FAISS + BM25)", value=True, help="Combines Dense FAISS vector similarity and Sparse BM25 keyword matching.")
    use_mmr = st.sidebar.toggle("✨ Maximal Marginal Relevance (MMR)", value=True, help="Ensures result diversity across different documents and pages.")
    top_k = st.sidebar.slider("Top-K Retrieved Chunks", min_value=1, max_value=10, value=5)

    st.sidebar.subheader("Chunking Settings")
    chunk_size = st.sidebar.select_slider("Chunk Size (chars)", options=[400, 500, 700, 1000], value=700)
    chunk_overlap = st.sidebar.select_slider("Chunk Overlap (chars)", options=[50, 100, 120, 200], value=120)

    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

    rag = get_pipeline(chunk_size, chunk_overlap, embedding_model, use_hybrid, use_mmr)
    rag.llm_engine.provider = llm_provider
    if api_key:
        rag.llm_engine.api_key = api_key

    stats = rag.get_stats()

    # Sidebar Document Ingestion Management
    st.sidebar.subheader("📁 Document Management")
    uploaded_files = st.sidebar.file_uploader("Upload additional PDFs", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        for uf in uploaded_files:
            save_path = os.path.join(DOCUMENTS_DIR, uf.name)
            with open(save_path, "wb") as f:
                f.write(uf.getbuffer())
        st.sidebar.success(f"Saved {len(uploaded_files)} new PDF(s) to '{DOCUMENTS_DIR}'!")
        if st.sidebar.button("🔄 Re-index Pipeline"):
            with st.spinner("Extracting text, deduplicating files, and building FAISS+BM25 indices..."):
                rag.ingest_documents(force_reindex=True)
                st.cache_resource.clear()
                st.rerun()

    if stats["total_chunks"] == 0:
        if st.sidebar.button("🚀 Ingest PDFs Now", type="primary"):
            with st.spinner("Extracting text, chunking, generating embeddings, and building FAISS+BM25 indices..."):
                stats = rag.ingest_documents(force_reindex=True)
                st.sidebar.success("Ingestion Complete!")
                st.rerun()

    # Main UI Header
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">PDF RAG Pipeline Dashboard (Production Edition)</h1>
        <div class="subtitle">PDFs ➔ Text Extraction & De-hyphenation ➔ Semantic Chunking ➔ Embeddings ➔ FAISS + BM25 Hybrid DB ➔ MMR Retrieval ➔ LLM ➔ Answer + Page Citations</div>
    </div>
    """, unsafe_allow_html=True)

    # Metrics Row
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.markdown(f'<div class="metric-card"><div class="metric-val">{stats.get("total_documents", 0)}</div><div class="metric-lbl">Unique PDFs</div></div>', unsafe_allow_html=True)
    m2.markdown(f'<div class="metric-card"><div class="metric-val">{stats.get("total_chunks", 0)}</div><div class="metric-lbl">Vector Chunks</div></div>', unsafe_allow_html=True)
    m3.markdown(f'<div class="metric-card"><div class="metric-val">FAISS+BM25</div><div class="metric-lbl">Hybrid Index</div></div>', unsafe_allow_html=True)
    m4.markdown(f'<div class="metric-card"><div class="metric-val">MMR</div><div class="metric-lbl">Diversification</div></div>', unsafe_allow_html=True)
    m5.markdown(f'<div class="metric-card"><div class="metric-val">100%</div><div class="metric-lbl">Page Citations</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Tabs
    tab_query, tab_explorer, tab_docs, tab_arch = st.tabs([
        "🎯 RAG Query & Citation System",
        "📊 Hybrid Vector Database Explorer",
        "📄 Indexed PDF Documents",
        "🏗️ Improved Pipeline Architecture"
    ])

    with tab_query:
        st.subheader("Ask Questions across Indexed PDF Documents")
        
        sample_queries = [
            "What deep learning models or geospatial analytics frameworks are used for land cover change detection?",
            "What is GeoFusion-ChangeNet and how does it encode features?",
            "How does feature-aware spectral learning improve temporal consistency in LULC mapping?",
            "What satellite remote sensing datasets were used in the research?"
        ]
        
        selected_sample = st.selectbox("💡 Select a sample query or enter your own:", ["-- Type custom query --"] + sample_queries)
        
        default_query_text = "" if selected_sample == "-- Type custom query --" else selected_sample
        user_query = st.text_input("Enter your query:", value=default_query_text, placeholder="e.g. How is land cover dynamics predicted in eastern region of India?")

        col_search, col_info = st.columns([4, 1])
        with col_search:
            btn_submit = st.button("🔎 Execute Hybrid RAG Pipeline", type="primary", use_container_width=True)

        if btn_submit and user_query.strip():
            with st.spinner("1️⃣ Query Embedding ➔ 2️⃣ Hybrid FAISS+BM25 Search (RRF) ➔ 3️⃣ MMR Diversification ➔ 4️⃣ Synthesizing Answer with Citations..."):
                t0 = time.time()
                result = rag.query(user_query, top_k=top_k)
                elapsed = time.time() - t0

            st.success(f"Query completed in {elapsed:.2f} seconds | Backend: **{result['provider']}** | Hybrid Search: **{'Enabled' if use_hybrid else 'Disabled'}** | MMR: **{'Enabled' if use_mmr else 'Disabled'}**")
            
            st.markdown("### 💬 Synthesized Answer")
            st.markdown(result["answer"])

            st.markdown("---")
            st.markdown("### 📚 Source Document & Page Citations")
            citations = result.get("citations", [])
            if citations:
                for c in citations:
                    st.markdown(f'<span class="citation-badge">📄 {c["citation"]}</span> <span class="score-badge">Similarity: {c["score"]:.4f}</span>', unsafe_allow_html=True)
            else:
                st.info("No citations found.")

            st.markdown("---")
            st.markdown(f"### 📑 Top-{top_k} Retrieved Chunks from Hybrid Vector Database")
            for idx, chunk in enumerate(result.get("retrieved_chunks", []), 1):
                rrf_info = f" | RRF Score: {chunk.get('rrf_score', 0.0):.4f}" if use_hybrid else ""
                with st.expander(f"Chunk #{idx} | [{chunk['doc_name']}, Page {chunk['page_number']}] | Dense Score: {chunk['similarity_score']:.4f}{rrf_info}"):
                    st.markdown(f"**Chunk ID:** `{chunk['chunk_id']}` | **Page:** {chunk['page_number']}/{chunk.get('total_pages', 1)}")
                    st.markdown(f'<div class="chunk-box">{chunk["text"]}</div>', unsafe_allow_html=True)

    with tab_explorer:
        st.subheader("Explore All Chunks in Hybrid Vector Database")
        if rag.vector_store.metadata:
            df = pd.DataFrame(rag.vector_store.metadata)
            search_filter = st.text_input("Filter chunks by text or document name:")
            if search_filter:
                df = df[df["text"].str.contains(search_filter, case=False, na=False) | df["doc_name"].str.contains(search_filter, case=False, na=False)]
            
            st.write(f"Showing **{len(df)}** of **{len(rag.vector_store.metadata)}** chunks:")
            st.dataframe(
                df[["doc_name", "page_number", "chunk_id", "char_count", "text"]],
                use_container_width=True,
                height=450
            )
        else:
            st.info("No vector chunks available. Please run document ingestion.")

    with tab_docs:
        st.subheader("Indexed PDF Documents (Deduplicated)")
        if stats.get("documents"):
            doc_list = stats["documents"]
            selected_doc = st.selectbox("Select PDF Document to inspect:", doc_list)
            
            doc_chunks = [c for c in rag.vector_store.metadata if c.get("doc_name") == selected_doc]
            st.write(f"Document **{selected_doc}** has **{len(doc_chunks)}** indexed chunks.")
            
            for c in doc_chunks[:5]:
                with st.expander(f"Page {c['page_number']} - Chunk {c['chunk_index']}"):
                    st.write(c["text"])
            if len(doc_chunks) > 5:
                st.write(f"...and {len(doc_chunks) - 5} more chunks.")
        else:
            st.info("No documents currently indexed.")

    with tab_arch:
        st.subheader("Improved Production Pipeline Architecture")
        st.markdown("""
        ```mermaid
        graph TD
            A["📄 PDF Documents (data/documents)"] --> B["🔒 SHA256 Deduplication & Text Cleaning"]
            B --> C["🧩 Semantic Paragraph Chunker"]
            C --> D["🔠 SentenceTransformer Dense Embeddings"]
            C --> E["🔤 BM25 Sparse Lexical Corpus"]
            D --> F["⚡ FAISS Dense Vector Index"]
            E --> G["⚡ BM25 Sparse Index"]
            H["❓ User Query"] --> I["🔠 Query Embedding"]
            H --> J["🔤 Query Tokens"]
            I --> F
            J --> G
            F --> K["🔀 Reciprocal Rank Fusion (RRF)"]
            G --> K
            K --> L["✨ Maximal Marginal Relevance (MMR)"]
            L --> M["🤖 Multi-LLM Engine (OpenAI / Ollama / Local)"]
            M --> N["📝 Synthesized Answer + Verified Citations"]
        ```
        """)

if __name__ == "__main__":
    main()

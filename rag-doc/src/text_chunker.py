import re
from typing import List, Dict, Any

class TextChunker:
    """
    Semantic Paragraph & Sentence-Aware Text Chunker:
    Recursively splits page text into overlapping chunks respecting structural boundaries.
    """

    def __init__(self, chunk_size: int = 700, chunk_overlap: int = 120):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_page(self, page_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split a single page's clean text into semantic overlapping chunks.
        """
        text = page_data.get("text", "").strip()
        if not text:
            return []

        doc_name = page_data["doc_name"]
        page_num = page_data["page_number"]
        total_pages = page_data.get("total_pages", 1)
        file_path = page_data.get("file_path", "")

        # Split page into structural paragraphs
        paragraphs = [p.strip() for p in re.split(r'\n\n+', text) if p.strip()]

        chunks = []
        current_chunk = ""
        chunk_idx = 0

        for para in paragraphs:
            # If paragraph itself fits within chunk size
            if len(current_chunk) + len(para) + 2 <= self.chunk_size:
                current_chunk = f"{current_chunk}\n\n{para}".strip()
            else:
                # If current_chunk has content, push it as a chunk
                if len(current_chunk) >= 40:
                    chunk_id = f"{doc_name}_p{page_num}_c{chunk_idx}"
                    citation = f"[{doc_name}, Page {page_num}]"
                    chunks.append({
                        "chunk_id": chunk_id,
                        "doc_name": doc_name,
                        "page_number": page_num,
                        "total_pages": total_pages,
                        "file_path": file_path,
                        "chunk_index": chunk_idx,
                        "text": current_chunk,
                        "char_count": len(current_chunk),
                        "citation": citation
                    })
                    chunk_idx += 1

                # If paragraph itself is longer than chunk_size, split by sentences
                if len(para) > self.chunk_size:
                    sentences = re.split(r'(?<=[.!?]) +', para)
                    sub_chunk = ""
                    for sent in sentences:
                        if len(sub_chunk) + len(sent) + 1 <= self.chunk_size:
                            sub_chunk = f"{sub_chunk} {sent}".strip()
                        else:
                            if len(sub_chunk) >= 40:
                                chunk_id = f"{doc_name}_p{page_num}_c{chunk_idx}"
                                citation = f"[{doc_name}, Page {page_num}]"
                                chunks.append({
                                    "chunk_id": chunk_id,
                                    "doc_name": doc_name,
                                    "page_number": page_num,
                                    "total_pages": total_pages,
                                    "file_path": file_path,
                                    "chunk_index": chunk_idx,
                                    "text": sub_chunk,
                                    "char_count": len(sub_chunk),
                                    "citation": citation
                                })
                                chunk_idx += 1
                                # Preserve overlap from tail of sub_chunk
                                overlap_text = sub_chunk[-self.chunk_overlap:] if len(sub_chunk) > self.chunk_overlap else ""
                                sub_chunk = f"{overlap_text} {sent}".strip()
                            else:
                                sub_chunk = sent
                    current_chunk = sub_chunk
                else:
                    # Carry over overlap text from previous paragraph for context continuity
                    overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else ""
                    current_chunk = f"{overlap_text}\n\n{para}".strip()

        # Emit final trailing chunk if any
        if len(current_chunk) >= 40:
            chunk_id = f"{doc_name}_p{page_num}_c{chunk_idx}"
            citation = f"[{doc_name}, Page {page_num}]"
            chunks.append({
                "chunk_id": chunk_id,
                "doc_name": doc_name,
                "page_number": page_num,
                "total_pages": total_pages,
                "file_path": file_path,
                "chunk_index": chunk_idx,
                "text": current_chunk,
                "char_count": len(current_chunk),
                "citation": citation
            })

        return chunks

    def chunk_all_pages(self, pages_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunks all extracted pages.
        """
        all_chunks = []
        for page in pages_data:
            chunks = self.chunk_page(page)
            all_chunks.extend(chunks)

        print(f"[TextChunker] Chunked {len(pages_data)} pages into {len(all_chunks)} semantic chunks (size={self.chunk_size}, overlap={self.chunk_overlap})")
        return all_chunks

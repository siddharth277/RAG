import os
import glob
import re
import hashlib
from typing import List, Dict, Any, Tuple

class PDFExtractor:
    """
    Advanced PDF Extractor with:
    - SHA256 File Deduplication
    - Automatic De-hyphenation & Line Cleaning
    - Page-level metadata preservation
    """

    def __init__(self, documents_dir: str = "data/documents", enable_dedup: bool = True):
        self.documents_dir = documents_dir
        self.enable_dedup = enable_dedup
        self.seen_hashes = {}  # sha256 -> doc_name

    def compute_file_hash(self, pdf_path: str) -> str:
        """Compute SHA-256 hash of a file to detect duplicates."""
        hasher = hashlib.sha256()
        with open(pdf_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def clean_text(self, raw_text: str) -> str:
        """
        Clean PDF extraction artifacts:
        - Fix broken line hyphenations (e.g. 'geo-\nspatial' -> 'geospatial')
        - Replace multiple spaces and weird unicode control characters
        - Normalize line breaks
        """
        if not raw_text:
            return ""

        # Fix hyphenation across line breaks: e.g. "frame-\nwork" -> "framework"
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', raw_text)
        
        # Replace standalone carriage returns or form feeds
        text = text.replace('\r', '\n').replace('\x0c', '\n')
        
        # Collapse multiple horizontal whitespaces
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Remove consecutive blank lines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()

    def extract_from_file(self, pdf_path: str) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Extract text from a single PDF file page by page.
        Returns (pages_data, is_duplicate).
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        doc_name = os.path.basename(pdf_path)

        if self.enable_dedup:
            file_hash = self.compute_file_hash(pdf_path)
            if file_hash in self.seen_hashes:
                original_doc = self.seen_hashes[file_hash]
                print(f"  - [Deduplication] '{doc_name}' is identical to '{original_doc}'. Skipping duplicate.")
                return [], True
            self.seen_hashes[file_hash] = doc_name

        pages_data = []

        # PyMuPDF (fitz) extraction
        try:
            import pymupdf as fitz
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            for page_idx in range(total_pages):
                page = doc[page_idx]
                raw_text = page.get_text("text")
                text = self.clean_text(raw_text)
                if text:
                    pages_data.append({
                        "doc_name": doc_name,
                        "file_path": pdf_path,
                        "page_number": page_idx + 1,
                        "total_pages": total_pages,
                        "text": text
                    })
            doc.close()
            return pages_data, False
        except ImportError:
            pass
        except Exception as e:
            print(f"[Warning] PyMuPDF failed on {doc_name}: {e}. Falling back to pypdf...")

        # pypdf fallback
        try:
            import pypdf
            reader = pypdf.PdfReader(pdf_path)
            total_pages = len(reader.pages)
            for page_idx, page in enumerate(reader.pages):
                raw_text = page.extract_text() or ""
                text = self.clean_text(raw_text)
                if text:
                    pages_data.append({
                        "doc_name": doc_name,
                        "file_path": pdf_path,
                        "page_number": page_idx + 1,
                        "total_pages": total_pages,
                        "text": text
                    })
            return pages_data, False
        except Exception as e:
            print(f"[Error] Failed to extract text from {doc_name}: {e}")
            return [], False

    def extract_all(self, directory: str = None) -> List[Dict[str, Any]]:
        """
        Extract text from all PDF files in target directory with automatic deduplication.
        """
        target_dir = directory or self.documents_dir
        pdf_paths = sorted(glob.glob(os.path.join(target_dir, "*.pdf")))
        
        all_pages = []
        skipped_count = 0
        print(f"[PDFExtractor] Found {len(pdf_paths)} PDF files in '{target_dir}'")

        for path in pdf_paths:
            pages, is_dup = self.extract_from_file(path)
            if is_dup:
                skipped_count += 1
            else:
                all_pages.extend(pages)
                print(f"  - {os.path.basename(path)}: extracted {len(pages)} pages")

        print(f"[PDFExtractor] Extraction complete: {len(all_pages)} total pages extracted ({skipped_count} duplicate files skipped).")
        return all_pages

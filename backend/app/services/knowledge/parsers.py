"""Document Parsers for PDF, TXT/MD, CSV, and Excel with page/location tracking."""
import io
import csv
from pathlib import Path
from typing import Dict, List, Tuple
import pymupdf  # PyMuPDF
import openpyxl
from backend.app.core.logging import logger


class DocumentParser:
    """Multi-format parser extracting text, tables, and page metadata safely."""

    @staticmethod
    def parse_pdf(file_bytes: bytes) -> Tuple[str, int, List[Dict[str, int]]]:
        """Parses PDF bytes into full text and per-page sections.
        
        Returns:
            (full_text, pages_count, list_of_page_dicts)
        """
        doc = pymupdf.open(stream=file_bytes, filetype="pdf")
        pages_count = len(doc)
        full_text_parts = []
        page_chunks = []

        for page_idx in range(pages_count):
            page_num = page_idx + 1
            page = doc.load_page(page_idx)
            text = page.get_text("text").strip()
            if text:
                full_text_parts.append(f"--- [Page {page_num}] ---\n{text}")
                page_chunks.append({
                    "page": page_num,
                    "text": text,
                })

        doc.close()
        full_text = "\n\n".join(full_text_parts)
        return full_text, pages_count, page_chunks

    @staticmethod
    def parse_text(file_bytes: bytes, encoding: str = "utf-8") -> Tuple[str, int, List[Dict[str, int]]]:
        """Parses plain text or Markdown files."""
        try:
            content = file_bytes.decode(encoding)
        except UnicodeDecodeError:
            content = file_bytes.decode("latin-1", errors="replace")

        full_text = content.strip()
        page_chunks = [{"page": 1, "text": full_text}]
        return full_text, 1, page_chunks

    @staticmethod
    def parse_csv(file_bytes: bytes, encoding: str = "utf-8") -> Tuple[str, int, List[Dict[str, int]]]:
        """Parses CSV tabular data into structured lines."""
        try:
            text_data = file_bytes.decode(encoding)
        except UnicodeDecodeError:
            text_data = file_bytes.decode("latin-1", errors="replace")

        reader = csv.reader(io.StringIO(text_data))
        rows = list(reader)
        lines = []
        for i, row in enumerate(rows):
            lines.append(f"Row {i+1}: " + " | ".join(row))

        full_text = "\n".join(lines)
        page_chunks = [{"page": 1, "text": full_text}]
        return full_text, 1, page_chunks

    @staticmethod
    def parse_excel(file_bytes: bytes) -> Tuple[str, int, List[Dict[str, int]]]:
        """Parses Excel (.xlsx) workbooks across all sheets."""
        wb = openpyxl.load_workbook(io.BytesIO(file_bytes), data_only=True)
        sheet_texts = []
        page_chunks = []

        for page_num, sheet_name in enumerate(wb.sheetnames, start=1):
            sheet = wb[sheet_name]
            lines = [f"=== Sheet: {sheet_name} ==="]
            for row in sheet.iter_rows(values_only=True):
                non_empty = [str(val).strip() for val in row if val is not None and str(val).strip()]
                if non_empty:
                    lines.append(" | ".join(non_empty))
            sheet_text = "\n".join(lines)
            sheet_texts.append(sheet_text)
            page_chunks.append({"page": page_num, "sheet": sheet_name, "text": sheet_text})

        wb.close()
        full_text = "\n\n".join(sheet_texts)
        return full_text, len(wb.sheetnames), page_chunks

    @staticmethod
    def parse_docx(file_bytes: bytes) -> Tuple[str, int, List[Dict[str, Any]]]:
        """Parses DOCX documents using standard zipfile and xml parsing."""
        import zipfile
        import xml.etree.ElementTree as ET

        try:
            with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
                xml_content = z.read("word/document.xml")
                tree = ET.fromstring(xml_content)
                paragraphs = []
                for p in tree.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p"):
                    texts = [elem.text for elem in p.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t") if elem.text]
                    if texts:
                        paragraphs.append("".join(texts).strip())
                full_text = "\n".join([p for p in paragraphs if p])
                page_chunks = [{"page": 1, "text": full_text}]
                return full_text, 1, page_chunks
        except Exception as e:
            logger.error(f"Failed to parse DOCX: {e}")
            return "", 0, []

    @classmethod
    def parse_file(cls, file_name: str, file_bytes: bytes) -> Tuple[str, int, List[Dict[str, int]]]:
        """Auto-detects file format by extension and parses."""
        ext = Path(file_name).suffix.lower()
        if ext == ".pdf":
            return cls.parse_pdf(file_bytes)
        elif ext in [".txt", ".md", ".markdown"]:
            return cls.parse_text(file_bytes)
        elif ext == ".csv":
            return cls.parse_csv(file_bytes)
        elif ext in [".xlsx", ".xls"]:
            return cls.parse_excel(file_bytes)
        elif ext in [".docx", ".doc"]:
            return cls.parse_docx(file_bytes)
        else:
            # Fallback to plain text
            return cls.parse_text(file_bytes)


"""Document Parsers for PDF, TXT/MD, CSV, and Excel with page/location tracking."""
import io
import csv
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple
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
            images = page.get_images(full=True)
            if text:
                full_text_parts.append(f"--- [Page {page_num}] ---\n{text}")
                page_chunks.append({
                    "page": page_num,
                    "location_type": "page",
                    "location_reference": f"Page {page_num}",
                    "text": text,
                })
            else:
                zero_text_desc = f"[Page {page_num}: Visual / Image Only - No Extractable Text ({len(images)} images)]"
                full_text_parts.append(f"--- [Page {page_num}] ---\n{zero_text_desc}")
                page_chunks.append({
                    "page": page_num,
                    "location_type": "page",
                    "location_reference": f"Page {page_num}",
                    "text": zero_text_desc,
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
        """Parses DOCX documents using standard zipfile and xml parsing with session and table awareness."""
        import zipfile
        import xml.etree.ElementTree as ET

        try:
            with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
                xml_content = z.read("word/document.xml")
                tree = ET.fromstring(xml_content)
                ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

                # Check for tables
                tables = tree.findall(".//w:tbl", ns)
                paragraphs = []
                for p in tree.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p"):
                    texts = [elem.text for elem in p.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t") if elem.text]
                    if texts:
                        paragraphs.append("".join(texts).strip())

                full_text = "\n".join([p for p in paragraphs if p])

                page_chunks: List[Dict[str, Any]] = []

                # Detect session markers (Order Flow sessions)
                session_markers = []
                for idx, p in enumerate(paragraphs):
                    if ("جلسه" in p and any(str(n) in p for n in range(11, 26))) or (p.startswith("جلسه ") or p.startswith("جلسه\u200c")):
                        session_markers.append((idx, p))

                if session_markers:
                    for s_idx, (p_start, s_title) in enumerate(session_markers):
                        p_end = session_markers[s_idx + 1][0] if s_idx + 1 < len(session_markers) else len(paragraphs)
                        session_text = "\n".join(paragraphs[p_start:p_end]).strip()
                        session_num_match = re.search(r"جلسه\s*(\d+)", s_title)
                        session_num = int(session_num_match.group(1)) if session_num_match else s_idx + 1
                        page_chunks.append({
                            "page": session_num,
                            "location_type": "session",
                            "location_reference": s_title[:100],
                            "title": s_title[:255],
                            "text": session_text,
                        })
                elif tables:
                    row_chunks = []
                    row_idx = 1
                    for tbl in tables:
                        for tr in tbl.findall(".//w:tr", ns):
                            cells = []
                            for tc in tr.findall(".//w:tc", ns):
                                cell_texts = [t.text for t in tc.findall(".//w:t", ns) if t.text]
                                cells.append("".join(cell_texts).strip())
                            row_str = " | ".join([c for c in cells if c])
                            if row_str:
                                row_chunks.append({
                                    "page": 1,
                                    "location_type": "table_row",
                                    "location_reference": f"Row {row_idx}",
                                    "title": f"Row {row_idx}: {row_str[:80]}",
                                    "text": row_str,
                                })
                                row_idx += 1
                    if row_chunks:
                        page_chunks = row_chunks
                    else:
                        page_chunks = [{"page": 1, "location_type": "page", "location_reference": "Page 1", "text": full_text}]
                else:
                    page_chunks = [{"page": 1, "location_type": "page", "location_reference": "Page 1", "text": full_text}]

                chunks_count = len(page_chunks) if page_chunks else 1
                return full_text, chunks_count, page_chunks
        except Exception as e:
            logger.error(f"Failed to parse DOCX: {e}")
            return "", 0, []

    @staticmethod
    def parse_zip(file_bytes: bytes, max_chunk_chars: int = 8000) -> Tuple[str, int, List[Dict[str, Any]]]:
        """Parses ZIP archives into bounded structured member chunks without silent truncation."""
        import zipfile

        try:
            with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
                infolist = [info for info in z.infolist() if not info.is_dir()]
                page_chunks: List[Dict[str, Any]] = []
                full_text_parts: List[str] = []

                # Supported text/code/data extensions
                valid_exts = {".yaml", ".yml", ".json", ".md", ".txt", ".py"}
                for info in infolist:
                    fn = info.filename
                    ext = Path(fn).suffix.lower()
                    if ext not in valid_exts:
                        continue

                    try:
                        raw_data = z.read(fn)
                        text = raw_data.decode("utf-8")
                    except UnicodeDecodeError:
                        text = raw_data.decode("latin-1", errors="replace")

                    if not text.strip():
                        continue

                    full_text_parts.append(f"=== File: {fn} ===\n{text}")
                    total_chars = len(text)

                    if total_chars <= max_chunk_chars:
                        page_chunks.append({
                            "page": 1,
                            "location_type": "archive_member",
                            "location_reference": fn,
                            "title": f"Archive Member: {fn}",
                            "text": text,
                        })
                    else:
                        # Bounded Structured Chunking with explicit coverage bounds
                        start = 0
                        part_num = 1
                        total_parts = (total_chars + max_chunk_chars - 1) // max_chunk_chars
                        while start < total_chars:
                            end = min(start + max_chunk_chars, total_chars)
                            chunk_subtext = text[start:end]
                            page_chunks.append({
                                "page": part_num,
                                "location_type": "archive_member",
                                "location_reference": f"{fn} [Part {part_num}/{total_parts}]",
                                "title": f"Archive Member: {fn} (Part {part_num}/{total_parts})",
                                "text": f"[Part {part_num}/{total_parts}, chars {start}-{end} of {total_chars}]\n{chunk_subtext}",
                            })
                            start = end
                            part_num += 1

                full_text = "\n\n".join(full_text_parts)
                return full_text, len(page_chunks), page_chunks
        except Exception as e:
            logger.error(f"Failed to parse ZIP: {e}")
            return "", 0, []

    @classmethod
    def parse_file(cls, file_name: str, file_bytes: bytes) -> Tuple[str, int, List[Dict[str, Any]]]:
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
        elif ext == ".zip":
            return cls.parse_zip(file_bytes)
        else:
            # Fallback to plain text
            return cls.parse_text(file_bytes)


"""Document Parsers for PDF, TXT/MD, CSV, and Excel with page/location tracking."""
import io
import csv
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple
import pymupdf  # PyMuPDF
import openpyxl
from backend.app.core.logging import logger

# Per-member decompression cap for archive formats (ZIP strategy bundles and
# DOCX containers). File-level caps apply to the compressed archive, but a
# crafted member can expand ~1000x on read (decompression bomb): bound the
# DECOMPRESSED bytes of every member before they reach decode/fromstring.
# Real strategy/text members are orders of magnitude below this.
_MAX_MEMBER_BYTES = 64 * 1024 * 1024

# Strict session-heading pattern (Order Flow جزوه، جلسات 11-25): a heading
# must START with "جلسه" + a 1-2 digit (Persian or ASCII) number, an optional
# separator, and a title. Anchored with bounded quantifiers, so body-text
# sentences that merely MENTION a session never match. Module-level constant:
# the regression tests must exercise THIS object, not a copy-pasted lookalike
# that cannot detect production drift.
SESSION_HEADING_PATTERN = re.compile(
    r"^\s*جلسه\s*([0-9۱۲۳۴۵۶۷۸۹۰]{1,2})(?:\s*([\|:—\-\.]|\s)\s*(.*))?$"
)


class DocumentParser:
    """Multi-format parser extracting text, tables, and page metadata safely."""

    @staticmethod
    def parse_pdf(file_bytes: bytes) -> Tuple[str, int, List[Dict[str, Any]]]:
        """Parses PDF bytes into full text and per-page sections.

        Categorizes pages into:
          - TEXT_VECTOR_PAGE: contains extractable text
          - IMAGE_ONLY_PAGE: visual/scanned page with images but no extractable text
          - VECTOR_ONLY_PAGE: visual vector drawing page with no extractable text or raster images
          - BLANK_PAGE: page with no text, images, or drawings

        Returns:
            (full_text, pages_count, list_of_page_dicts)
        """
        return DocumentParser.parse_pdf_path(io.BytesIO(file_bytes))

    @staticmethod
    def parse_pdf_path(source) -> Tuple[str, int, List[Dict[str, Any]]]:
        """Parses a PDF from a filesystem path or binary stream (both are
        accepted by pymupdf.open), so the raw bytes need not be materialized."""
        from backend.app.core.config import settings

        if isinstance(source, (str, Path)):
            doc = pymupdf.open(filename=str(source), filetype="pdf")
        else:
            doc = pymupdf.open(stream=source.read(), filetype="pdf")
        pages_count = len(doc)
        full_text_parts = []
        page_chunks = []

        if settings.OCR_ENABLED:
            logger.debug("OCR_ENABLED is active (feature flag enabled, no external OCR executed)")

        for page_idx in range(pages_count):
            page_num = page_idx + 1
            page = doc.load_page(page_idx)
            text = page.get_text("text").strip()
            images = page.get_images(full=True)
            drawings = page.get_drawings()

            image_details = []
            for img in images:
                xref = img[0]
                width = img[2] if len(img) > 2 else None
                height = img[3] if len(img) > 3 else None
                image_details.append({
                    "xref": xref,
                    "width": width,
                    "height": height,
                })

            if text:
                page_type = "TEXT_VECTOR_PAGE"
                is_semantic = True
                desc = text
                full_text_parts.append(f"--- [Page {page_num}] ---\n{text}")
            elif images:
                page_type = "IMAGE_ONLY_PAGE"
                is_semantic = False
                dims_summary = ", ".join(
                    f"{img['width']}x{img['height']}"
                    for img in image_details
                    if img.get("width") and img.get("height")
                )
                dims_str = f" [{dims_summary}]" if dims_summary else ""
                desc = f"[Page {page_num}: Visual / Image Only - No Extractable Text ({len(images)} images{dims_str})]"
                full_text_parts.append(f"--- [Page {page_num}] ---\n{desc}")
            elif drawings:
                page_type = "VECTOR_ONLY_PAGE"
                is_semantic = False
                desc = f"[Page {page_num}: Vector Graphics Only - No Extractable Text ({len(drawings)} vector paths)]"
                full_text_parts.append(f"--- [Page {page_num}] ---\n{desc}")
            else:
                page_type = "BLANK_PAGE"
                is_semantic = False
                desc = f"[Page {page_num}: Blank Page / Visual / Image Only - No Extractable Text or Graphics]"
                full_text_parts.append(f"--- [Page {page_num}] ---\n{desc}")

            page_chunks.append({
                "page": page_num,
                "location_type": "page",
                "location_reference": f"Page {page_num}",
                "title": f"Page {page_num}",
                "text": desc,
                "page_type": page_type,
                "is_semantic_content": is_semantic,
                "image_count": len(images),
                "image_dimensions": image_details,
                "parameters": {
                    "page_type": page_type,
                    "is_semantic_content": is_semantic,
                    "image_count": len(images),
                    "image_dimensions": image_details,
                },
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
    def parse_text_path(path: Path, encoding: str = "utf-8") -> Tuple[str, int, List[Dict[str, int]]]:
        """Reads a text file as text (never as one bytes object), keeping the
        same decode fallback as parse_text. Note the extracted text itself is
        stored as Document.raw_text, so text content does reach memory by design."""
        with open(path, "r", encoding=encoding, errors="strict") as f:
            try:
                content = f.read()
            except UnicodeDecodeError:
                with open(path, "r", encoding="latin-1", errors="replace") as f2:
                    content = f2.read()
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
        return DocumentParser._csv_from_text(text_data)

    @staticmethod
    def parse_csv_path(path: Path, encoding: str = "utf-8") -> Tuple[str, int, List[Dict[str, int]]]:
        """Reads a CSV file as text (never as one bytes object)."""
        with open(path, "r", encoding=encoding, newline="") as f:
            try:
                text_data = f.read()
            except UnicodeDecodeError:
                with open(path, "r", encoding="latin-1", errors="replace", newline="") as f2:
                    text_data = f2.read()
        return DocumentParser._csv_from_text(text_data)

    @staticmethod
    def _csv_from_text(text_data: str) -> Tuple[str, int, List[Dict[str, int]]]:
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
        return DocumentParser._excel_from_workbook(wb)

    @staticmethod
    def parse_excel_path(path: Path) -> Tuple[str, int, List[Dict[str, int]]]:
        """Parses an Excel workbook from disk without materializing its bytes."""
        wb = openpyxl.load_workbook(str(path), data_only=True)
        return DocumentParser._excel_from_workbook(wb)

    @staticmethod
    def _excel_from_workbook(wb) -> Tuple[str, int, List[Dict[str, int]]]:
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

        sheet_count = len(wb.sheetnames)
        wb.close()
        full_text = "\n\n".join(sheet_texts)
        return full_text, sheet_count, page_chunks

    @staticmethod
    def parse_docx(file_bytes: bytes) -> Tuple[str, int, List[Dict[str, Any]]]:
        """Parses DOCX documents using zipfile and hardened XML parsing (no DOCTYPE/ENTITY)."""
        import zipfile
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
            return DocumentParser.parse_docx_from_zip(z)

    @staticmethod
    def parse_docx_path(path: Path) -> Tuple[str, int, List[Dict[str, Any]]]:
        """Parses a DOCX file from disk without materializing its bytes."""
        import zipfile
        with zipfile.ZipFile(str(path)) as z:
            return DocumentParser.parse_docx_from_zip(z)

    @staticmethod
    def parse_docx_from_zip(z) -> Tuple[str, int, List[Dict[str, Any]]]:
        """Shared DOCX body over an open ZipFile; defusedxml forbids DTDs/entities."""
        from defusedxml import ElementTree as DT

        try:
            if True:
                # Bounded member read: word/document.xml can be crafted to
                # expand far beyond the archive's file-level size cap.
                with z.open("word/document.xml") as member:
                    xml_content = member.read(_MAX_MEMBER_BYTES + 1)
                if len(xml_content) > _MAX_MEMBER_BYTES:
                    raise ValueError(
                        "DOCX member 'word/document.xml' exceeds the "
                        f"{_MAX_MEMBER_BYTES}-byte per-member limit."
                    )
                # defusedxml forbids DTDs/entities entirely: a DOCTYPE raises before expansion.
                tree = DT.fromstring(xml_content)
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

                # Detect session markers (Order Flow sessions 11 to 25) with the
                # module-level strict pattern (kept in sync with its tests).
                session_pattern = SESSION_HEADING_PATTERN
                session_markers = []
                for idx, p in enumerate(paragraphs):
                    if len(p) <= 120:
                        m = session_pattern.match(p)
                        if m:
                            num_str = m.group(1).translate(str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789"))
                            session_num = int(num_str)
                            if 1 <= session_num <= 50:
                                session_markers.append((idx, p, session_num))

                if session_markers:
                    # Optional introductory section before the first session
                    first_p_start = session_markers[0][0]
                    if first_p_start > 0:
                        intro_text = "\n".join(paragraphs[:first_p_start]).strip()
                        if intro_text:
                            page_chunks.append({
                                "page": 0,
                                "location_type": "section",
                                "location_reference": "Intro",
                                "title": "مقدمه و راهنمای جزوه",
                                "text": intro_text,
                                "parameters": {
                                    "is_semantic_content": True,
                                    "paragraph_start_1based": 1,
                                    "paragraph_end_1based": first_p_start,
                                },
                            })

                    for s_idx, (p_start, s_title, session_num) in enumerate(session_markers):
                        p_end = session_markers[s_idx + 1][0] if s_idx + 1 < len(session_markers) else len(paragraphs)
                        session_text = "\n".join(paragraphs[p_start:p_end]).strip()
                        loc_ref = f"Session {session_num}: {s_title[:80]}" if s_title else f"Session {session_num}"
                        page_chunks.append({
                            "page": session_num,
                            "location_type": "session",
                            "location_reference": loc_ref,
                            "title": s_title[:255],
                            "text": session_text,
                            "parameters": {
                                "session_number": session_num,
                                "paragraph_start_1based": p_start + 1,
                                "paragraph_end_1based": p_end,
                                "is_semantic_content": True,
                            },
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
        except Exception:
            # A corrupt/unreadable source must surface as a FAILED ingest, not
            # as an empty success: returning ("", 0, []) here is indistinguishable
            # from a legitimately blank file and becomes a TERMINAL
            # NO_EXTRACTABLE_CONTENT record (silent data loss). The manager's
            # parse-except path marks the document failed with the real cause.
            raise

    @staticmethod
    def parse_zip(file_bytes: bytes, max_chunk_chars: int = 8000) -> Tuple[str, int, List[Dict[str, Any]]]:
        """Parses ZIP archives into bounded structured member chunks without silent truncation."""
        import zipfile
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
            return DocumentParser.parse_zip_from_zip(z, max_chunk_chars)

    @staticmethod
    def parse_zip_path(path: Path, max_chunk_chars: int = 8000) -> Tuple[str, int, List[Dict[str, Any]]]:
        """Parses a ZIP archive from disk without materializing its bytes."""
        import zipfile
        with zipfile.ZipFile(str(path)) as z:
            return DocumentParser.parse_zip_from_zip(z, max_chunk_chars)

    @staticmethod
    def parse_zip_from_zip(z, max_chunk_chars: int = 8000) -> Tuple[str, int, List[Dict[str, Any]]]:
        """Shared ZIP body over an open ZipFile."""
        try:
            if True:
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
                        # Bounded per-member read (decompression-bomb guard):
                        # read at most cap+1 bytes, then verify.
                        with z.open(fn) as member:
                            raw_data = member.read(_MAX_MEMBER_BYTES + 1)
                    except ValueError:
                        raise
                    if len(raw_data) > _MAX_MEMBER_BYTES:
                        raise ValueError(
                            f"ZIP member '{fn}' exceeds the {_MAX_MEMBER_BYTES}-byte "
                            "per-member limit; refusing possible decompression bomb."
                        )
                    try:
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
        except Exception:
            # Same fail-loud contract as parse_docx: a parse failure must not
            # become a terminal empty-success record (see parse_docx comment).
            raise

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

    @classmethod
    def parse_file_path(cls, file_name: str, file_path: Path) -> Tuple[str, int, List[Dict[str, Any]]]:
        """Auto-detects file format by extension and parses from disk.

        PDF, XLSX, DOCX, and ZIP are handed to their libraries as paths, so the
        raw file is never held as one bytes object. Text/CSV are read as text
        (str), which the ingestion contract already requires because the
        extracted text is persisted as Document.raw_text.
        """
        ext = Path(file_name).suffix.lower()
        if ext == ".pdf":
            return cls.parse_pdf_path(file_path)
        elif ext in [".txt", ".md", ".markdown"]:
            return cls.parse_text_path(file_path)
        elif ext == ".csv":
            return cls.parse_csv_path(file_path)
        elif ext in [".xlsx", ".xls"]:
            return cls.parse_excel_path(file_path)
        elif ext in [".docx", ".doc"]:
            return cls.parse_docx_path(file_path)
        elif ext == ".zip":
            return cls.parse_zip_path(file_path)
        else:
            # Fallback to plain text
            return cls.parse_text_path(file_path)


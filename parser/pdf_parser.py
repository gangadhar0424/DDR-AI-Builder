"""
PDF Parser Module.

Extracts text, headings, tables, and structural metadata from PDF files
using PyMuPDF (fitz). Designed for inspection and thermal report parsing
with font-aware heading detection and page-level segmentation.
"""

import re
from pathlib import Path
from dataclasses import dataclass, field

import fitz  # PyMuPDF
from loguru import logger

import config


@dataclass
class TextBlock:
    """A block of text extracted from a PDF page."""

    text: str
    font_size: float
    font_name: str
    is_bold: bool
    bbox: tuple[float, float, float, float]  # x0, y0, x1, y1


@dataclass
class ParsedPage:
    """Structured representation of a single parsed PDF page."""

    page_number: int
    raw_text: str
    headings: list[str] = field(default_factory=list)
    text_blocks: list[TextBlock] = field(default_factory=list)
    tables: list[list[list[str]]] = field(default_factory=list)
    image_count: int = 0


@dataclass
class ParsedDocument:
    """Complete parsed PDF document."""

    file_path: str
    file_name: str
    total_pages: int
    pages: list[ParsedPage] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        """Concatenate all page text into a single string."""
        return "\n\n".join(
            f"--- Page {p.page_number} ---\n{p.raw_text}"
            for p in self.pages
            if p.raw_text.strip()
        )

    @property
    def all_headings(self) -> list[tuple[int, str]]:
        """Return (page_number, heading) tuples across all pages."""
        results = []
        for page in self.pages:
            for heading in page.headings:
                results.append((page.page_number, heading))
        return results


class PDFParser:
    """
    Production-grade PDF parser using PyMuPDF.

    Extracts text with font metadata, detects headings based on
    font size thresholds and bold styling, and counts embedded images
    per page for downstream cross-referencing.
    """

    def __init__(
        self,
        heading_font_threshold: float | None = None,
    ):
        """
        Initialize the PDF parser.

        Args:
            heading_font_threshold: Minimum font size to classify a line
                as a heading. Defaults to config value.
        """
        self.heading_font_threshold = (
            heading_font_threshold or config.HEADING_FONT_SIZE_THRESHOLD
        )

    def parse(self, pdf_path: str | Path) -> ParsedDocument:
        """
        Parse a PDF file into a structured ParsedDocument.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            ParsedDocument containing all extracted data.

        Raises:
            FileNotFoundError: If the PDF path does not exist.
            RuntimeError: If the PDF cannot be opened or parsed.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logger.info(f"Parsing PDF: {pdf_path.name}")

        try:
            doc = fitz.open(str(pdf_path))
        except Exception as e:
            raise RuntimeError(f"Failed to open PDF '{pdf_path.name}': {e}") from e

        parsed = ParsedDocument(
            file_path=str(pdf_path),
            file_name=pdf_path.name,
            total_pages=len(doc),
            metadata=self._extract_metadata(doc),
        )

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            parsed_page = self._parse_page(page, page_idx + 1)
            parsed.pages.append(parsed_page)

        doc.close()

        logger.info(
            f"Parsed {parsed.total_pages} pages from '{pdf_path.name}' — "
            f"{sum(len(p.headings) for p in parsed.pages)} headings detected"
        )
        return parsed

    def _extract_metadata(self, doc: fitz.Document) -> dict:
        """Extract document-level metadata."""
        meta = doc.metadata or {}
        return {
            "title": meta.get("title", ""),
            "author": meta.get("author", ""),
            "subject": meta.get("subject", ""),
            "creator": meta.get("creator", ""),
            "producer": meta.get("producer", ""),
            "creation_date": meta.get("creationDate", ""),
            "modification_date": meta.get("modDate", ""),
        }

    def _parse_page(self, page: fitz.Page, page_number: int) -> ParsedPage:
        """
        Parse a single PDF page.

        Args:
            page: PyMuPDF page object.
            page_number: 1-indexed page number.

        Returns:
            ParsedPage with extracted text, headings, text blocks, and image count.
        """
        raw_text = page.get_text("text") or ""
        text_blocks = self._extract_text_blocks(page)
        headings = self._detect_headings(text_blocks)
        tables = self._extract_tables(page)
        image_count = len(page.get_images(full=True))

        return ParsedPage(
            page_number=page_number,
            raw_text=raw_text.strip(),
            headings=headings,
            text_blocks=text_blocks,
            tables=tables,
            image_count=image_count,
        )

    def _extract_text_blocks(self, page: fitz.Page) -> list[TextBlock]:
        """
        Extract text blocks with font metadata from a page.

        Uses PyMuPDF's 'dict' text extraction to get per-span font info.
        """
        blocks = []
        try:
            text_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
        except Exception:
            logger.warning(f"Failed dict extraction on page {page.number + 1}")
            return blocks

        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:  # type 0 = text block
                continue
            for line in block.get("lines", []):
                line_text_parts = []
                max_font_size = 0.0
                font_name = ""
                is_bold = False

                for span in line.get("spans", []):
                    span_text = span.get("text", "").strip()
                    if not span_text:
                        continue
                    line_text_parts.append(span_text)
                    size = span.get("size", 0)
                    if size > max_font_size:
                        max_font_size = size
                        font_name = span.get("font", "")
                    # Detect bold: check font flags or font name
                    flags = span.get("flags", 0)
                    if flags & 2**4:  # bit 4 = bold
                        is_bold = True
                    elif "bold" in span.get("font", "").lower():
                        is_bold = True

                line_text = " ".join(line_text_parts).strip()
                if line_text:
                    bbox = line.get("bbox", (0, 0, 0, 0))
                    blocks.append(
                        TextBlock(
                            text=line_text,
                            font_size=max_font_size,
                            font_name=font_name,
                            is_bold=is_bold,
                            bbox=tuple(bbox),
                        )
                    )

        return blocks

    def _detect_headings(self, text_blocks: list[TextBlock]) -> list[str]:
        """
        Identify headings based on font size and bold styling.

        A text block is classified as a heading if:
        - Its font size exceeds the heading threshold, OR
        - It is bold AND its text length is under 120 characters
          (to avoid bold paragraphs being misclassified).
        """
        headings = []
        for block in text_blocks:
            text = block.text.strip()
            if not text or len(text) < 2:
                continue

            is_heading = False

            # Large font → heading
            if block.font_size >= self.heading_font_threshold:
                is_heading = True

            # Bold + short text → heading
            if block.is_bold and len(text) <= 120:
                is_heading = True

            # Common heading patterns (numbered sections, etc.)
            if re.match(
                r"^(\d+\.?\s+|[A-Z]\.\s+|Section\s+\d+|Chapter\s+\d+)",
                text,
                re.IGNORECASE,
            ):
                if len(text) <= 120:
                    is_heading = True

            if is_heading:
                # Clean up heading text
                cleaned = re.sub(r"\s+", " ", text).strip()
                if cleaned and cleaned not in headings:
                    headings.append(cleaned)

        return headings

    def _extract_tables(self, page: fitz.Page) -> list[list[list[str]]]:
        """
        Attempt basic table extraction from a page.

        Uses PyMuPDF's built-in table finder when available,
        falls back to empty list if not supported.
        """
        tables = []
        try:
            # PyMuPDF >= 1.23.0 has find_tables()
            if hasattr(page, "find_tables"):
                found = page.find_tables()
                for table in found:
                    extracted = table.extract()
                    if extracted:
                        # Clean cell values
                        cleaned = [
                            [
                                (cell.strip() if cell else "")
                                for cell in row
                            ]
                            for row in extracted
                        ]
                        tables.append(cleaned)
        except Exception as e:
            logger.debug(f"Table extraction skipped on page {page.number + 1}: {e}")

        return tables


def parse_pdf(pdf_path: str | Path) -> ParsedDocument:
    """
    Convenience function for one-shot PDF parsing.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        ParsedDocument with all extracted content.
    """
    parser = PDFParser()
    return parser.parse(pdf_path)

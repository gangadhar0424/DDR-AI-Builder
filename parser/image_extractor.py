"""
Image Extractor Module.

Extracts embedded images from PDF files, filters by quality thresholds,
and saves them with metadata for downstream mapping to DDR sections.
"""

import io
from pathlib import Path
from dataclasses import dataclass, field

import fitz  # PyMuPDF
from loguru import logger

import config


@dataclass
class ExtractedImage:
    """Metadata and path for an image extracted from a PDF."""

    image_path: str
    page_number: int
    image_index: int
    width: int
    height: int
    colorspace: str
    file_size_bytes: int
    nearby_text: str = ""  # text found near the image on the same page


@dataclass
class ImageExtractionResult:
    """Result of image extraction from a complete PDF."""

    source_pdf: str
    total_images_found: int
    total_images_saved: int
    images: list[ExtractedImage] = field(default_factory=list)
    skipped_reasons: list[str] = field(default_factory=list)


class ImageExtractor:
    """
    Extracts and saves images from PDF documents.

    Applies size/quality filters and captures nearby text for each image
    to enable intelligent image-to-section mapping in the DDR.
    """

    def __init__(
        self,
        output_dir: str | Path | None = None,
        min_width: int | None = None,
        min_height: int | None = None,
        max_size_mb: float | None = None,
    ):
        """
        Initialize the image extractor.

        Args:
            output_dir: Directory to save extracted images.
            min_width: Minimum image width in pixels.
            min_height: Minimum image height in pixels.
            max_size_mb: Maximum image file size in MB.
        """
        self.output_dir = Path(output_dir or config.TEMP_IMAGE_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.min_width = min_width or config.MIN_IMAGE_WIDTH
        self.min_height = min_height or config.MIN_IMAGE_HEIGHT
        self.max_size_bytes = int((max_size_mb or config.MAX_IMAGE_SIZE_MB) * 1024 * 1024)

    def extract(self, pdf_path: str | Path) -> ImageExtractionResult:
        """
        Extract all qualifying images from a PDF.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            ImageExtractionResult with saved image paths and metadata.

        Raises:
            FileNotFoundError: If the PDF does not exist.
            RuntimeError: If the PDF cannot be opened.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logger.info(f"Extracting images from: {pdf_path.name}")

        try:
            doc = fitz.open(str(pdf_path))
        except Exception as e:
            raise RuntimeError(f"Failed to open PDF: {e}") from e

        result = ImageExtractionResult(
            source_pdf=str(pdf_path),
            total_images_found=0,
            total_images_saved=0,
        )

        # Create a subdirectory per PDF
        pdf_image_dir = self.output_dir / pdf_path.stem
        pdf_image_dir.mkdir(parents=True, exist_ok=True)

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_number = page_idx + 1
            images = page.get_images(full=True)
            result.total_images_found += len(images)

            for img_idx, img_info in enumerate(images):
                xref = img_info[0]
                extracted = self._extract_single_image(
                    doc, page, xref, page_number, img_idx, pdf_image_dir
                )
                if extracted:
                    result.images.append(extracted)
                    result.total_images_saved += 1
                else:
                    result.skipped_reasons.append(
                        f"Page {page_number}, image {img_idx}: "
                        f"skipped (below quality threshold)"
                    )

        doc.close()

        logger.info(
            f"Extracted {result.total_images_saved}/{result.total_images_found} "
            f"images from '{pdf_path.name}'"
        )
        return result

    def _extract_single_image(
        self,
        doc: fitz.Document,
        page: fitz.Page,
        xref: int,
        page_number: int,
        img_idx: int,
        output_dir: Path,
    ) -> ExtractedImage | None:
        """
        Extract and save a single image if it meets quality thresholds.

        Args:
            doc: The open PyMuPDF document.
            page: The page containing the image.
            xref: Cross-reference ID of the image in the PDF.
            page_number: 1-indexed page number.
            img_idx: Index of the image on the page.
            output_dir: Directory to save the image file.

        Returns:
            ExtractedImage if saved, None if filtered out.
        """
        try:
            base_image = doc.extract_image(xref)
            if not base_image:
                return None

            image_bytes = base_image["image"]
            width = base_image.get("width", 0)
            height = base_image.get("height", 0)
            ext = base_image.get("ext", "png")
            colorspace_val = base_image.get("colorspace", 0)

            # Determine colorspace name
            if isinstance(colorspace_val, int):
                cs_map = {1: "Gray", 3: "RGB", 4: "CMYK"}
                colorspace = cs_map.get(colorspace_val, f"CS({colorspace_val})")
            else:
                colorspace = str(colorspace_val)

            # Apply quality filters
            if width < self.min_width or height < self.min_height:
                return None
            if len(image_bytes) > self.max_size_bytes:
                return None
            if len(image_bytes) < 500:  # Skip tiny decorative images
                return None

            # Save the image
            filename = f"page{page_number}_img{img_idx}.{ext}"
            save_path = output_dir / filename
            save_path.write_bytes(image_bytes)

            # Capture nearby text for context
            nearby_text = self._get_nearby_text(page, xref)

            return ExtractedImage(
                image_path=str(save_path),
                page_number=page_number,
                image_index=img_idx,
                width=width,
                height=height,
                colorspace=colorspace,
                file_size_bytes=len(image_bytes),
                nearby_text=nearby_text,
            )

        except Exception as e:
            logger.warning(
                f"Failed extracting image xref={xref} on page {page_number}: {e}"
            )
            return None

    def _get_nearby_text(self, page: fitz.Page, xref: int) -> str:
        """
        Get text near an image on its page for contextual mapping.

        Extracts the full page text as a rough proxy. For more precision,
        one could compute image bounding boxes and find text blocks
        within a radius — but full-page text is sufficient for LLM matching.
        """
        try:
            text = page.get_text("text") or ""
            # Truncate to avoid oversized context
            return text[:1500].strip()
        except Exception:
            return ""


def extract_images(pdf_path: str | Path) -> ImageExtractionResult:
    """
    Convenience function for one-shot image extraction.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        ImageExtractionResult with all extracted image metadata.
    """
    extractor = ImageExtractor()
    return extractor.extract(pdf_path)

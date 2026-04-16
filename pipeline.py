"""
DDR-AI-Builder Pipeline Orchestrator.

Coordinates the end-to-end pipeline:
  PDF Parsing → Observation Extraction → Merging → Conflict Detection →
  Missing Data Handling → DDR Generation → Export

Can be used programmatically or via the Streamlit UI.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict

from loguru import logger

import config
from parser.pdf_parser import PDFParser, ParsedDocument
from parser.image_extractor import ImageExtractor, ImageExtractionResult
from extraction.observation_extractor import ObservationExtractor
from extraction.thermal_extractor import ThermalExtractor
from processing.merger import ObservationMerger, MergeResult
from processing.conflict_detector import ConflictDetector, ConflictReport
from processing.missing_data_handler import MissingDataHandler, MissingDataReport
from generation.ddr_generator import DDRGenerator, DDRReport


@dataclass
class PipelineResult:
    """Complete result from the DDR pipeline execution."""

    success: bool = False
    error: str = ""
    elapsed_seconds: float = 0.0

    # Intermediate results
    inspection_parsed: ParsedDocument | None = None
    thermal_parsed: ParsedDocument | None = None
    inspection_images: ImageExtractionResult | None = None
    thermal_images: ImageExtractionResult | None = None
    inspection_observations: list = field(default_factory=list)
    thermal_observations: list = field(default_factory=list)
    merge_result: MergeResult | None = None
    conflict_report: ConflictReport | None = None
    missing_report: MissingDataReport | None = None

    # Final outputs
    ddr_report: DDRReport | None = None
    html_path: str = ""
    pdf_path: str = ""
    markdown_path: str = ""

    @property
    def summary(self) -> str:
        """Human-readable pipeline summary."""
        if not self.success:
            return f"Pipeline failed: {self.error}"
        parts = [
            f"✅ DDR generated successfully in {self.elapsed_seconds:.1f}s",
            f"   Inspection observations: {len(self.inspection_observations)}",
            f"   Thermal observations: {len(self.thermal_observations)}",
        ]
        if self.merge_result:
            parts.append(f"   Merged observations: {self.merge_result.total_merged}")
            parts.append(f"   Corroborated: {self.merge_result.corroborated_count}")
            parts.append(f"   Areas: {len(self.merge_result.areas)}")
        if self.conflict_report:
            parts.append(f"   Conflicts detected: {self.conflict_report.total_conflicts}")
        if self.missing_report:
            parts.append(f"   Data completeness: {self.missing_report.completeness_score:.0%}")
        if self.html_path:
            parts.append(f"   HTML: {self.html_path}")
        if self.pdf_path:
            parts.append(f"   PDF:  {self.pdf_path}")
        if self.markdown_path:
            parts.append(f"   MD:   {self.markdown_path}")
        return "\n".join(parts)


class DDRPipeline:
    """
    End-to-end DDR generation pipeline.

    Orchestrates all phases from PDF parsing through final export,
    with progress callbacks for UI integration.
    """

    def __init__(self, progress_callback=None):
        """
        Initialize the pipeline.

        Args:
            progress_callback: Optional callable(step: str, progress: float)
                for UI progress reporting. progress is 0.0 to 1.0.
        """
        self.progress = progress_callback or (lambda s, p: None)
        self.pdf_parser = PDFParser()
        self.image_extractor = ImageExtractor()
        self.observation_extractor = ObservationExtractor()
        self.thermal_extractor = ThermalExtractor()
        self.merger = ObservationMerger()
        self.conflict_detector = ConflictDetector()
        self.missing_handler = MissingDataHandler()
        self.ddr_generator = DDRGenerator()
        self.debug_dir = config.OUTPUT_DIR / "debug"

    def _save_debug(self, filename: str, data) -> None:
        """Save intermediate artifact to outputs/debug/ for traceability."""
        try:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            path = self.debug_dir / filename

            if hasattr(data, "to_dict"):
                serializable = data.to_dict()
            elif hasattr(data, "__dataclass_fields__"):
                serializable = asdict(data)
            elif isinstance(data, list):
                serializable = [
                    item.to_dict() if hasattr(item, "to_dict")
                    else asdict(item) if hasattr(item, "__dataclass_fields__")
                    else item
                    for item in data
                ]
            else:
                serializable = data

            path.write_text(
                json.dumps(serializable, indent=2, default=str, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.debug(f"Debug artifact saved: {filename}")
        except Exception as e:
            logger.warning(f"Failed to save debug artifact {filename}: {e}")

    def run(
        self,
        inspection_pdf: str | Path,
        thermal_pdf: str | Path,
        report_title: str | None = None,
        export_formats: list[str] | None = None,
    ) -> PipelineResult:
        """
        Execute the complete DDR pipeline.

        Args:
            inspection_pdf: Path to inspection report PDF.
            thermal_pdf: Path to thermal report PDF.
            report_title: Custom title for the DDR report.
            export_formats: List of formats to export. Options: "html", "pdf", "markdown".
                Defaults to ["html", "markdown"].

        Returns:
            PipelineResult with all intermediate and final outputs.
        """
        if export_formats is None:
            export_formats = ["html", "markdown"]

        result = PipelineResult()
        start_time = time.time()

        try:
            # ── Phase 1: Validate configuration ──
            self.progress("Validating configuration...", 0.0)
            errors = config.validate_config()
            if errors:
                result.error = " | ".join(errors)
                logger.error(f"Configuration validation failed: {result.error}")
                return result

            # ── Phase 2: Parse PDFs ──
            self.progress("Parsing inspection PDF...", 0.05)
            result.inspection_parsed = self.pdf_parser.parse(inspection_pdf)
            self._save_debug("parsed_inspection_raw.json", {
                "filename": result.inspection_parsed.file_name,
                "total_pages": result.inspection_parsed.total_pages,
                "headings_count": len(result.inspection_parsed.all_headings),
                "headings": result.inspection_parsed.all_headings[:50],
            })

            self.progress("Parsing thermal PDF...", 0.12)
            result.thermal_parsed = self.pdf_parser.parse(thermal_pdf)
            self._save_debug("parsed_thermal_raw.json", {
                "filename": result.thermal_parsed.file_name,
                "total_pages": result.thermal_parsed.total_pages,
                "headings_count": len(result.thermal_parsed.all_headings),
                "headings": result.thermal_parsed.all_headings[:50],
            })

            # ── Phase 3: Extract images ──
            self.progress("Extracting images from inspection PDF...", 0.18)
            result.inspection_images = self.image_extractor.extract(inspection_pdf)

            self.progress("Extracting images from thermal PDF...", 0.22)
            result.thermal_images = self.image_extractor.extract(thermal_pdf)

            # Combine all images
            all_images = []
            if result.inspection_images:
                all_images.extend(result.inspection_images.images)
            if result.thermal_images:
                all_images.extend(result.thermal_images.images)

            # ── Phase 4: Extract observations ──
            self.progress("Extracting inspection observations (LLM)...", 0.28)
            result.inspection_observations = self.observation_extractor.extract(
                result.inspection_parsed
            )
            self._save_debug("extracted_inspection_observations.json",
                             result.inspection_observations)

            self.progress("Extracting thermal observations (LLM)...", 0.40)
            result.thermal_observations = self.thermal_extractor.extract(
                result.thermal_parsed
            )
            self._save_debug("extracted_thermal_observations.json",
                             result.thermal_observations)

            # ── Phase 5: Merge observations ──
            self.progress("Merging observations (semantic matching)...", 0.55)
            result.merge_result = self.merger.merge(
                result.inspection_observations,
                result.thermal_observations,
            )
            self._save_debug("merged_observations.json", result.merge_result)

            # ── Phase 6: Detect conflicts ──
            self.progress("Detecting conflicts...", 0.62)
            result.conflict_report = self.conflict_detector.detect(
                result.merge_result.merged_observations
            )
            self._save_debug("conflict_analysis.json", result.conflict_report)

            # ── Phase 7: Handle missing data ──
            self.progress("Checking data completeness...", 0.67)
            result.merge_result, result.missing_report = (
                self.missing_handler.process(result.merge_result)
            )

            # ── Phase 8: Generate DDR ──
            self.progress("Generating DDR report (LLM)...", 0.72)
            result.ddr_report = self.ddr_generator.generate(
                merge_result=result.merge_result,
                conflict_report=result.conflict_report,
                missing_report=result.missing_report,
                images=all_images,
                report_title=report_title,
            )
            self._save_debug("final_ddr_structured.json", result.ddr_report)

            # ── Phase 9: Export ──
            if "html" in export_formats:
                self.progress("Exporting HTML...", 0.90)
                result.html_path = self.ddr_generator.export_html(result.ddr_report)

            if "pdf" in export_formats and result.html_path:
                self.progress("Exporting PDF...", 0.93)
                result.pdf_path = self.ddr_generator.export_pdf(result.html_path)

            if "markdown" in export_formats:
                self.progress("Exporting Markdown...", 0.96)
                result.markdown_path = self.ddr_generator.export_markdown(
                    result.ddr_report
                )

            result.success = True
            self.progress("Done!", 1.0)

        except FileNotFoundError as e:
            result.error = f"File not found: {e}"
            logger.error(result.error)
        except Exception as e:
            result.error = f"Pipeline error: {e}"
            logger.exception("Pipeline failed")
        finally:
            result.elapsed_seconds = time.time() - start_time

        return result


def run_pipeline(
    inspection_pdf: str,
    thermal_pdf: str,
    report_title: str | None = None,
) -> PipelineResult:
    """
    Convenience function to run the full pipeline.

    Args:
        inspection_pdf: Path to inspection report PDF.
        thermal_pdf: Path to thermal report PDF.
        report_title: Custom report title.

    Returns:
        PipelineResult with all outputs.
    """
    pipeline = DDRPipeline()
    return pipeline.run(inspection_pdf, thermal_pdf, report_title)


# ═══════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="DDR-AI-Builder: Generate Detailed Diagnostic Reports from PDF inputs."
    )
    parser.add_argument(
        "inspection_pdf",
        help="Path to the inspection report PDF.",
    )
    parser.add_argument(
        "thermal_pdf",
        help="Path to the thermal report PDF.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Custom report title.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["html", "markdown"],
        choices=["html", "pdf", "markdown"],
        help="Export formats (default: html markdown).",
    )

    args = parser.parse_args()

    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}")

    print("=" * 60)
    print("  DDR-AI-Builder Pipeline")
    print("=" * 60)
    print(f"  Inspection: {args.inspection_pdf}")
    print(f"  Thermal:    {args.thermal_pdf}")
    print(f"  Formats:    {', '.join(args.formats)}")
    print("=" * 60)

    pipeline = DDRPipeline(
        progress_callback=lambda step, pct: print(f"  [{pct:5.0%}] {step}")
    )
    result = pipeline.run(
        inspection_pdf=args.inspection_pdf,
        thermal_pdf=args.thermal_pdf,
        report_title=args.title,
        export_formats=args.formats,
    )

    print("\n" + result.summary)

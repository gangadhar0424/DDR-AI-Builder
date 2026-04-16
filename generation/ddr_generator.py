"""
DDR Generator Module.

Generates the final Detailed Diagnostic Report (DDR) by:
1. Using LLM to synthesize merged observations into professional prose
2. Rendering an HTML report via Jinja2 templates
3. Exporting to PDF (via WeasyPrint) and Markdown formats
4. Mapping extracted images to relevant DDR sections
"""

from __future__ import annotations

import os
import re
import base64
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict

from jinja2 import Environment, FileSystemLoader
from loguru import logger

import config
from llm_client import call_llm, call_llm_json
from processing.merger import MergeResult, MergedObservation
from processing.conflict_detector import ConflictReport
from processing.missing_data_handler import MissingDataReport
from parser.image_extractor import ExtractedImage


@dataclass
class DDRSection:
    """A single section of the DDR report."""

    title: str
    content: str
    observations: list[dict] = field(default_factory=list)
    images: list[dict] = field(default_factory=list)


@dataclass
class DDRReport:
    """Complete Detailed Diagnostic Report."""

    title: str
    generated_date: str
    property_summary: str
    area_observations: list[DDRSection] = field(default_factory=list)
    root_causes: str = ""
    severity_assessment: str = ""
    recommended_actions: str = ""
    additional_notes: str = ""
    missing_info: str = ""
    conflicts: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


DDR_SYNTHESIS_SYSTEM = """You are a professional building diagnostics engineer writing a
Detailed Diagnostic Report (DDR). Write in clear, professional, technical language.

RULES:
- Base ALL statements strictly on the provided observations.
- NEVER invent, assume, or hallucinate any facts.
- Include source references (document name and page numbers) for traceability.
- Use precise technical terminology.
- Structure content with clear headings and bullet points where appropriate.
- When severity is assessed, provide clear reasoning.
"""

PROPERTY_SUMMARY_PROMPT = """Based on the following merged observations from inspection and thermal reports,
write a concise Property Issue Summary (2-3 paragraphs).

Total observations: {total_obs}
Corroborated (found in both reports): {corroborated}
Areas affected: {areas}

Key findings by severity:
{severity_summary}

Write a professional executive summary covering:
1. Overall property condition assessment
2. Number and nature of issues found
3. Critical items requiring immediate attention

Return ONLY the summary text, no JSON.
"""

ROOT_CAUSE_PROMPT = """Based on the following observations, identify probable root causes.

OBSERVATIONS:
{observations_text}

For each distinct root cause:
1. State the probable cause clearly
2. List which observations it explains
3. Note any contributing factors
4. Rate your confidence (based solely on available evidence)

Write in professional report format. Return ONLY the analysis text.
"""

SEVERITY_PROMPT = """Assess the severity of each area based on these observations.

OBSERVATIONS:
{observations_text}

For each area, provide:
1. Overall severity rating (Critical / High / Medium / Low / Informational)
2. Clear reasoning for the rating
3. Risk factors considered
4. Timeline recommendation (immediate / 30 days / 90 days / routine maintenance)

Write in professional report format with clear reasoning. Return ONLY the assessment text.
"""

ACTIONS_PROMPT = """Based on these observations and their severity, provide recommended actions.

OBSERVATIONS:
{observations_text}

Structure recommendations as:
1. IMMEDIATE ACTIONS (safety and critical items)
2. SHORT-TERM ACTIONS (within 30 days)
3. MEDIUM-TERM ACTIONS (within 90 days)
4. LONG-TERM / MAINTENANCE ACTIONS

For each action:
- Specify what needs to be done
- Which area/observation it addresses
- Estimated priority
- Any prerequisites or dependencies

Write in professional report format. Return ONLY the recommendations text.
"""

AREA_ANALYSIS_PROMPT = """Write a detailed diagnostic analysis for the following area.

AREA: {area}

OBSERVATIONS:
{observations_text}

Write a professional analysis covering:
1. Current condition summary
2. Specific findings (reference source documents and pages)
3. Thermal data interpretation (if available)
4. Significance of findings

Keep it factual and traceable. Return ONLY the analysis text.
"""


class DDRGenerator:
    """
    Generates professional DDR reports from merged observations.

    Orchestrates LLM calls for each DDR section, maps images to
    relevant areas, and renders the final output via Jinja2 templates.
    """

    def __init__(self):
        """Initialize the DDR generator."""
        self.template_dir = config.TEMPLATE_DIR
        self.output_dir = config.OUTPUT_DIR

    def generate(
        self,
        merge_result: MergeResult,
        conflict_report: ConflictReport,
        missing_report: MissingDataReport,
        images: list[ExtractedImage] | None = None,
        report_title: str | None = None,
    ) -> DDRReport:
        """
        Generate a complete DDR report.

        Args:
            merge_result: Merged observations from both reports.
            conflict_report: Detected conflicts between reports.
            missing_report: Missing data analysis.
            images: Extracted images from both PDFs.
            report_title: Custom title for the report.

        Returns:
            DDRReport with all sections populated.
        """
        logger.info("Generating DDR report...")

        observations = merge_result.merged_observations
        report = DDRReport(
            title=report_title or config.DEFAULT_REPORT_TITLE,
            generated_date=datetime.now().strftime("%B %d, %Y at %I:%M %p"),
            property_summary="",
            metadata={
                "total_observations": merge_result.total_merged,
                "corroborated": merge_result.corroborated_count,
                "inspection_count": merge_result.total_inspection_obs,
                "thermal_count": merge_result.total_thermal_obs,
                "areas_count": len(merge_result.areas),
                "conflicts_count": conflict_report.total_conflicts,
                "completeness": f"{missing_report.completeness_score:.0%}",
            },
        )

        # 1. Property Summary
        logger.info("Generating property summary...")
        report.property_summary = self._generate_property_summary(
            merge_result
        )

        # 2. Area-wise Observations
        logger.info("Generating area-wise observations...")
        report.area_observations = self._generate_area_sections(
            merge_result, images
        )

        # 3. Root Causes
        logger.info("Generating root cause analysis...")
        report.root_causes = self._generate_root_causes(observations)

        # 4. Severity Assessment
        logger.info("Generating severity assessment...")
        report.severity_assessment = self._generate_severity_assessment(
            observations
        )

        # 5. Recommended Actions
        logger.info("Generating recommended actions...")
        report.recommended_actions = self._generate_actions(observations)

        # 6. Additional Notes
        report.additional_notes = self._generate_additional_notes(
            conflict_report, merge_result
        )

        # 7. Missing Information
        report.missing_info = self._format_missing_info(missing_report)

        # 8. Conflicts
        if conflict_report.has_conflicts():
            report.conflicts = [asdict(c) for c in conflict_report.conflicts]

        logger.info("DDR report generation complete")
        return report

    def _generate_property_summary(self, merge_result: MergeResult) -> str:
        """Generate the executive property summary."""
        obs = merge_result.merged_observations

        # Build severity summary
        severity_counts = {}
        for o in obs:
            sev = o.severity.lower() if o.severity else "unrated"
            severity_counts[sev] = severity_counts.get(sev, 0) + 1

        severity_lines = []
        for sev in ["critical", "high", "medium", "low", "informational", "unrated"]:
            count = severity_counts.get(sev, 0)
            if count > 0:
                severity_lines.append(f"  - {sev.title()}: {count} observations")

        prompt = PROPERTY_SUMMARY_PROMPT.format(
            total_obs=merge_result.total_merged,
            corroborated=merge_result.corroborated_count,
            areas=", ".join(merge_result.areas[:20]),
            severity_summary="\n".join(severity_lines),
        )

        try:
            return call_llm(prompt, system_prompt=DDR_SYNTHESIS_SYSTEM)
        except Exception as e:
            logger.error(f"Failed to generate property summary: {e}")
            return (
                f"This report covers {merge_result.total_merged} observations "
                f"across {len(merge_result.areas)} areas. "
                f"{merge_result.corroborated_count} findings were corroborated "
                f"by both inspection and thermal reports."
            )

    def _generate_area_sections(
        self,
        merge_result: MergeResult,
        images: list[ExtractedImage] | None,
    ) -> list[DDRSection]:
        """Generate detailed sections for each area."""
        sections = []
        areas = merge_result.areas

        for area in areas:
            area_obs = merge_result.get_by_area(area)
            if not area_obs:
                continue

            # Format observations for the prompt
            obs_text = self._format_observations(area_obs)

            # Generate analysis via LLM
            try:
                prompt = AREA_ANALYSIS_PROMPT.format(
                    area=area,
                    observations_text=obs_text,
                )
                analysis = call_llm(prompt, system_prompt=DDR_SYNTHESIS_SYSTEM)
            except Exception as e:
                logger.error(f"Failed area analysis for '{area}': {e}")
                analysis = obs_text

            # Map images to this area
            area_images = self._map_images_to_area(area, area_obs, images)

            section = DDRSection(
                title=area,
                content=analysis,
                observations=[o.to_dict() for o in area_obs],
                images=area_images,
            )
            sections.append(section)

        return sections

    def _generate_root_causes(
        self, observations: list[MergedObservation]
    ) -> str:
        """Generate root cause analysis."""
        obs_text = self._format_observations(observations[:30])
        prompt = ROOT_CAUSE_PROMPT.format(observations_text=obs_text)

        try:
            return call_llm(prompt, system_prompt=DDR_SYNTHESIS_SYSTEM)
        except Exception as e:
            logger.error(f"Root cause generation failed: {e}")
            return "Root cause analysis could not be generated. Please review observations manually."

    def _generate_severity_assessment(
        self, observations: list[MergedObservation]
    ) -> str:
        """Generate severity assessment with reasoning."""
        obs_text = self._format_observations(observations[:30])
        prompt = SEVERITY_PROMPT.format(observations_text=obs_text)

        try:
            return call_llm(prompt, system_prompt=DDR_SYNTHESIS_SYSTEM)
        except Exception as e:
            logger.error(f"Severity assessment failed: {e}")
            return "Severity assessment could not be generated. Please review observations manually."

    def _generate_actions(
        self, observations: list[MergedObservation]
    ) -> str:
        """Generate recommended actions."""
        obs_text = self._format_observations(observations[:30])
        prompt = ACTIONS_PROMPT.format(observations_text=obs_text)

        try:
            return call_llm(prompt, system_prompt=DDR_SYNTHESIS_SYSTEM)
        except Exception as e:
            logger.error(f"Actions generation failed: {e}")
            return "Action recommendations could not be generated. Please review observations manually."

    def _generate_additional_notes(
        self,
        conflict_report: ConflictReport,
        merge_result: MergeResult,
    ) -> str:
        """Generate additional notes section."""
        notes = []

        notes.append(
            f"This report was generated by DDR-AI-Builder on "
            f"{datetime.now().strftime('%B %d, %Y')}."
        )
        notes.append(
            f"Analysis is based on {merge_result.total_inspection_obs} "
            f"inspection observations and {merge_result.total_thermal_obs} "
            f"thermal observations."
        )

        if merge_result.corroborated_count > 0:
            notes.append(
                f"{merge_result.corroborated_count} findings were corroborated "
                f"by both reports, increasing diagnostic confidence."
            )

        if conflict_report.has_conflicts():
            notes.append(
                f"⚠ {conflict_report.total_conflicts} conflict(s) were detected "
                f"between inspection and thermal reports. These should be reviewed "
                f"by a qualified professional."
            )

        return "\n\n".join(notes)

    def _format_missing_info(self, missing_report: MissingDataReport) -> str:
        """Format missing information section."""
        if not missing_report.missing_fields:
            return "All required data fields are complete across all observations."

        lines = [
            f"Data completeness: {missing_report.completeness_score:.0%}\n"
        ]

        # Group by status
        by_status = {}
        for field in missing_report.missing_fields:
            by_status.setdefault(field.status, []).append(field)

        for status, fields in by_status.items():
            lines.append(f"\n**{status.title()} Fields ({len(fields)}):**")
            for f in fields[:20]:  # Cap to avoid excessive output
                lines.append(
                    f"- {f.area} → {f.field_name}: {f.description}"
                )

        return "\n".join(lines)

    def _format_observations(
        self, observations: list[MergedObservation]
    ) -> str:
        """Format a list of observations as text for LLM prompts."""
        parts = []
        for obs in observations:
            sources = ", ".join(obs.sources) if obs.sources else "Unknown"
            pages = ", ".join(obs.pages) if obs.pages else "N/A"
            corr = " [CORROBORATED]" if obs.is_corroborated else ""

            entry = (
                f"• Area: {obs.area}\n"
                f"  Finding: {obs.observation}\n"
                f"  Severity: {obs.severity}\n"
                f"  Sources: {sources} (Pages: {pages}){corr}\n"
                f"  Confidence: {obs.confidence_score:.0%}"
            )

            if obs.temperature_data:
                entry += f"\n  Temperature: {obs.temperature_data}"
            if obs.recommendation:
                entry += f"\n  Recommendation: {obs.recommendation}"

            parts.append(entry)

        return "\n\n".join(parts)

    def _map_images_to_area(
        self,
        area: str,
        area_obs: list[MergedObservation],
        images: list[ExtractedImage] | None,
    ) -> list[dict]:
        """
        Map extracted images to a specific area using multi-signal ranking.

        Ranking signals (weighted):
        1. Area/heading textual similarity (weight: 0.4)
        2. Caption/nearby-text match (weight: 0.3)
        3. Page proximity to observations (weight: 0.2)
        4. Image reference match (weight: 0.1)

        Falls back to "Image Mapping Uncertain" if confidence is low.
        """
        if not images:
            return []

        area_lower = area.lower()
        area_words = set(area_lower.split()) - {"the", "a", "an", "of", "in", "at", "on", "-"}

        # Collect observation page numbers and image references
        obs_pages = set()
        obs_image_refs = set()
        obs_keywords = set()
        for obs in area_obs:
            for page_ref in obs.pages:
                # Extract numeric page from "source p.X" format
                import re
                nums = re.findall(r"\d+", page_ref)
                obs_pages.update(int(n) for n in nums)
            for ref in obs.image_references:
                obs_image_refs.add(ref.lower())
            # Extract key terms from observation text
            for word in obs.observation.lower().split():
                if len(word) > 3:
                    obs_keywords.add(word)

        scored_images = []
        for img in images:
            score = 0.0
            nearby = (img.nearby_text or "").lower()

            # Signal 1: Area name in nearby text (0.4)
            if area_words:
                overlap = sum(1 for w in area_words if w in nearby)
                area_score = min(1.0, overlap / max(len(area_words), 1))
                score += area_score * 0.4

            # Signal 2: Caption/keyword similarity (0.3)
            if obs_keywords and nearby:
                keyword_hits = sum(1 for kw in list(obs_keywords)[:20] if kw in nearby)
                caption_score = min(1.0, keyword_hits / 5)
                score += caption_score * 0.3

            # Signal 3: Page proximity (0.2)
            if obs_pages:
                min_distance = min(abs(img.page_number - p) for p in obs_pages)
                page_score = max(0.0, 1.0 - (min_distance / 5))
                score += page_score * 0.2

            # Signal 4: Direct image reference match (0.1)
            if obs_image_refs:
                for ref in obs_image_refs:
                    if ref in nearby:
                        score += 0.1
                        break

            if score > 0.15:  # Minimum threshold
                mapping_label = "Confident" if score >= 0.5 else "Uncertain"
                img_data = {
                    "path": img.image_path,
                    "page": img.page_number,
                    "width": img.width,
                    "height": img.height,
                    "base64": self._image_to_base64(img.image_path),
                    "mapping_confidence": round(score, 2),
                    "mapping_label": mapping_label,
                }
                scored_images.append((score, img_data))

        # Sort by score descending and return top 5
        scored_images.sort(key=lambda x: x[0], reverse=True)
        return [img_data for _, img_data in scored_images[:5]]

    def _image_to_base64(self, image_path: str) -> str:
        """Convert an image file to base64 for HTML embedding."""
        try:
            path = Path(image_path)
            if not path.exists():
                return ""
            data = path.read_bytes()
            ext = path.suffix.lower().lstrip(".")
            mime = {
                "png": "image/png",
                "jpg": "image/jpeg",
                "jpeg": "image/jpeg",
                "gif": "image/gif",
                "bmp": "image/bmp",
            }.get(ext, "image/png")
            encoded = base64.b64encode(data).decode("utf-8")
            return f"data:{mime};base64,{encoded}"
        except Exception as e:
            logger.warning(f"Failed to encode image: {e}")
            return ""

    # ────────────────────────────────────────
    # Export Methods
    # ────────────────────────────────────────

    def export_html(
        self,
        report: DDRReport,
        output_path: str | Path | None = None,
    ) -> str:
        """
        Render the DDR as HTML using Jinja2 template.

        Args:
            report: Completed DDRReport.
            output_path: Output file path. Auto-generated if None.

        Returns:
            Path to the generated HTML file.
        """
        env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            autoescape=True,
        )

        template = env.get_template("ddr_template.html")
        html_content = template.render(
            report=report,
            severity_levels=config.SEVERITY_LEVELS,
            company_name=config.DEFAULT_COMPANY_NAME,
        )

        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"DDR_Report_{timestamp}.html"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html_content, encoding="utf-8")

        logger.info(f"HTML report saved: {output_path}")
        return str(output_path)

    def export_pdf(
        self,
        html_path: str | Path,
        output_path: str | Path | None = None,
    ) -> str:
        """
        Convert HTML report to PDF using WeasyPrint.

        Args:
            html_path: Path to the HTML report.
            output_path: Output PDF path. Auto-generated if None.

        Returns:
            Path to the generated PDF file.
        """
        try:
            from weasyprint import HTML
        except ImportError:
            logger.warning(
                "WeasyPrint not installed. Skipping PDF export. "
                "Install with: pip install weasyprint"
            )
            return ""

        html_path = Path(html_path)
        if output_path is None:
            output_path = html_path.with_suffix(".pdf")

        output_path = Path(output_path)

        try:
            HTML(filename=str(html_path)).write_pdf(str(output_path))
            logger.info(f"PDF report saved: {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"PDF export failed: {e}")
            return ""

    def export_markdown(
        self,
        report: DDRReport,
        output_path: str | Path | None = None,
    ) -> str:
        """
        Export DDR as a professional Markdown document.

        Includes source traceability, confidence scores, and structured
        section hierarchy suitable for client delivery.

        Args:
            report: Completed DDRReport.
            output_path: Output file path. Auto-generated if None.

        Returns:
            Path to the generated Markdown file.
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"DDR_Report_{timestamp}.md"

        output_path = Path(output_path)
        lines = []

        lines.append(f"# {report.title}")
        lines.append(f"\n*Generated: {report.generated_date}*")
        lines.append(f"*Report Version: {config.REPORT_VERSION} | "
                      f"Company: {config.DEFAULT_COMPANY_NAME}*\n")
        lines.append("---\n")

        # Metadata Summary Table
        if report.metadata:
            lines.append("## Report Summary\n")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            labels = {
                "total_observations": "Total Observations",
                "corroborated": "Corroborated Findings",
                "inspection_count": "Inspection Observations",
                "thermal_count": "Thermal Observations",
                "areas_count": "Areas Assessed",
                "conflicts_count": "Conflicts Detected",
                "completeness": "Data Completeness",
            }
            for key, label in labels.items():
                if key in report.metadata:
                    lines.append(f"| {label} | {report.metadata[key]} |")
            lines.append("")

        # Property Summary
        lines.append("\n## 1. Property Issue Summary\n")
        lines.append(report.property_summary)
        lines.append("")

        # Area Observations
        lines.append("\n## 2. Area-wise Diagnostic Observations\n")
        for section in report.area_observations:
            lines.append(f"\n### 📍 {section.title}\n")
            lines.append(section.content)

            if section.observations:
                lines.append("\n#### Detailed Findings\n")
                for obs in section.observations:
                    severity = obs.get('severity', 'N/A')
                    confidence = obs.get('confidence_score', 0)
                    corroborated = obs.get('is_corroborated', False)

                    # Confidence badge
                    if confidence >= 0.85:
                        conf_badge = "🟢 High"
                    elif confidence >= 0.65:
                        conf_badge = "🟡 Medium"
                    else:
                        conf_badge = "🔴 Low"

                    corr_tag = " ✅ *Corroborated*" if corroborated else ""

                    lines.append(
                        f"- **{obs.get('observation', 'N/A')}**\n"
                        f"  - Severity: `{severity}` | "
                        f"Confidence: {conf_badge} ({confidence:.0%}){corr_tag}"
                    )

                    # Source references
                    sources = obs.get('sources', [])
                    pages = obs.get('pages', [])
                    if sources or pages:
                        ref_parts = []
                        for p in pages:
                            ref_parts.append(p)
                        if not ref_parts and sources:
                            ref_parts = sources
                        lines.append(
                            f"  - 📄 Source: {', '.join(ref_parts)}"
                        )

                    # Temperature data
                    if obs.get('temperature_data'):
                        lines.append(
                            f"  - 🌡️ Temperature: {obs['temperature_data']}"
                        )

                    if obs.get('recommendation'):
                        lines.append(
                            f"  - 💡 Recommendation: {obs['recommendation']}"
                        )
            lines.append("")

        # Root Causes
        lines.append("\n## 3. Probable Root Cause Analysis\n")
        lines.append(report.root_causes)

        # Severity Assessment
        lines.append("\n\n## 4. Severity Assessment\n")
        lines.append(report.severity_assessment)

        # Recommended Actions
        lines.append("\n\n## 5. Recommended Actions\n")
        lines.append(report.recommended_actions)

        # Additional Notes
        lines.append("\n\n## 6. Additional Notes\n")
        lines.append(report.additional_notes)

        # Missing Info
        lines.append("\n\n## 7. Missing or Unclear Information\n")
        lines.append(report.missing_info)

        # Conflicts
        if report.conflicts:
            lines.append("\n\n## ⚠ Detected Conflicts Between Reports\n")
            for c in report.conflicts:
                lines.append(
                    f"- **{c.get('area', 'Unknown')}**: {c.get('description', '')}\n"
                    f"  Resolution: {c.get('resolution_suggestion', 'N/A')}"
                )

        # Footer
        lines.append("\n\n---\n")
        lines.append(
            f"*This report was generated by {config.DEFAULT_COMPANY_NAME}. "
            f"All findings are based on automated analysis of the provided inspection "
            f"and thermal reports and should be verified by a qualified professional.*"
        )

        content = "\n".join(lines)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")

        logger.info(f"Markdown report saved: {output_path}")
        return str(output_path)

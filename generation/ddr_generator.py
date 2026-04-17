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
class Introduction:
    background: str = "Not Available"
    objective: str = "Not Available"
    scope: str = "Not Available"
    tools_used: str = "Not Available"

@dataclass
class GeneralInfo:
    client_details: str = "Not Available"
    location: str = "Not Available"
    structure_type: str = "Not Available"
    age_of_building: str = "Not Available"
    floors: str = "Not Available"
    previous_repairs: str = "Not Available"

@dataclass
class DDRSection:
    """A single section of the DDR report."""
    title: str
    content: str
    thermal_findings: str = "Not Available"
    combined_interpretation: str = "Not Available"
    negative_inputs: list[str] = field(default_factory=list)
    positive_inputs: list[str] = field(default_factory=list)
    observations: list[dict] = field(default_factory=list)
    images: list[dict] = field(default_factory=list)


@dataclass
class DDRReport:
    """Complete Detailed Diagnostic Report."""
    title: str
    generated_date: str
    introduction: Introduction = field(default_factory=Introduction)
    general_info: GeneralInfo = field(default_factory=GeneralInfo)
    area_observations: list[DDRSection] = field(default_factory=list)
    root_causes: str = ""
    severity_assessment: str = ""
    recommended_actions: str = ""
    summary_table: list[dict] = field(default_factory=list)
    thermal_image_refs: list[dict] = field(default_factory=list)
    visual_image_refs: list[dict] = field(default_factory=list)
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

INTRO_INFO_PROMPT = """Based on the following observations, extract the Introduction details.
If any field cannot be inferred, write "Not Available".

OBSERVATIONS:
{observations_text}

Return a JSON object with these exact keys:
{{
  "background": "<background of inspection>",
  "objective": "<objective of assessment>",
  "scope": "<scope of work>",
  "tools_used": "<tools used>"
}}
"""

GENERAL_INFO_PROMPT = """Based on the following observations, extract the General Information details.
If any field cannot be inferred, write "Not Available".

OBSERVATIONS:
{observations_text}

Return a JSON object with these exact keys:
{{
  "client_details": "<client details>",
  "location": "<location>",
  "structure_type": "<structure type>",
  "age_of_building": "<age of building>",
  "floors": "<floors>",
  "previous_repairs": "<previous repairs>"
}}
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

SUMMARY_TABLE_PROMPT = """Based on the following observations, create a summary table of the issues.

OBSERVATIONS:
{observations_text}

Return a JSON array of objects. Each object must have these exact keys:
{{
  "issue": "<concise description of issue>",
  "affected_area": "<area affected>",
  "cause": "<probable cause>",
  "severity": "<Critical|High|Medium|Low|Informational>",
  "recommended_action": "<concise recommended action>"
}}
"""

AREA_ANALYSIS_PROMPT = """Write a detailed diagnostic analysis for the following area.

AREA: {area}

OBSERVATIONS:
{observations_text}

Return a JSON object with these exact keys:
{{
  "content": "<Current condition summary and specific visual findings>",
  "thermal_findings": "<Supporting thermal findings interpretation>",
  "combined_interpretation": "<Combined interpretation of visual and thermal data>",
  "negative_inputs": ["<issue observed 1>", "<issue observed 2>"],
  "positive_inputs": ["<source of issue 1>", "<source of issue 2>"]
}}
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

        # 1. Intro and General Info
        logger.info("Generating introduction and general info...")
        report.introduction, report.general_info = self._generate_intro_and_general_info(observations)

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

        # Summary Table
        logger.info("Generating summary table...")
        report.summary_table = self._generate_summary_table(observations)

        # Image References
        logger.info("Processing image references...")
        thermal_refs, visual_refs = self._process_image_references(images)
        report.thermal_image_refs = thermal_refs
        report.visual_image_refs = visual_refs

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

    def _generate_intro_and_general_info(self, observations: list[MergedObservation]) -> tuple[Introduction, GeneralInfo]:
        """Generate introduction and general information from observations."""
        obs_text = self._format_observations(observations[:30])
        intro = Introduction()
        gen_info = GeneralInfo()

        try:
            intro_json = call_llm_json(INTRO_INFO_PROMPT.format(observations_text=obs_text), system_prompt=DDR_SYNTHESIS_SYSTEM)
            if isinstance(intro_json, dict):
                intro.background = intro_json.get("background", "Not Available")
                intro.objective = intro_json.get("objective", "Not Available")
                intro.scope = intro_json.get("scope", "Not Available")
                intro.tools_used = intro_json.get("tools_used", "Not Available")
        except Exception as e:
            logger.error(f"Failed to generate intro info: {e}")

        try:
            gen_json = call_llm_json(GENERAL_INFO_PROMPT.format(observations_text=obs_text), system_prompt=DDR_SYNTHESIS_SYSTEM)
            if isinstance(gen_json, dict):
                gen_info.client_details = gen_json.get("client_details", "Not Available")
                gen_info.location = gen_json.get("location", "Not Available")
                gen_info.structure_type = gen_json.get("structure_type", "Not Available")
                gen_info.age_of_building = gen_json.get("age_of_building", "Not Available")
                gen_info.floors = gen_json.get("floors", "Not Available")
                gen_info.previous_repairs = gen_json.get("previous_repairs", "Not Available")
        except Exception as e:
            logger.error(f"Failed to generate general info: {e}")
            
        return intro, gen_info

    def _generate_summary_table(self, observations: list[MergedObservation]) -> list[dict]:
        """Generate the summary table."""
        obs_text = self._format_observations(observations[:40])
        prompt = SUMMARY_TABLE_PROMPT.format(observations_text=obs_text)
        
        try:
            res = call_llm_json(prompt, system_prompt=DDR_SYNTHESIS_SYSTEM)
            if isinstance(res, list):
                return res
            return []
        except Exception as e:
            logger.error(f"Failed to generate summary table: {e}")
            return []

    def _process_image_references(self, images: list[ExtractedImage] | None) -> tuple[list[dict], list[dict]]:
        """Process and split images into thermal and visual lists."""
        if not images:
            return [], []
            
        thermal_refs = []
        visual_refs = []
        
        for idx, img in enumerate(images):
            # If the original PDF was thermal or context suggests thermal
            is_thermal = False
            if "thermal" in str(img.image_path).lower():
                is_thermal = True
            elif img.nearby_text and any(k in img.nearby_text.lower() for k in ["ir", "thermal", "temperature", "flir", "infrared"]):
                is_thermal = True
                
            img_data = {
                "description": f"Image from page {img.page_number}",
                "base64": self._image_to_base64(img.image_path),
                "page": img.page_number,
                "source": "Report"
            }
            
            if is_thermal:
                thermal_refs.append(img_data)
            else:
                visual_refs.append(img_data)
                
        return thermal_refs[:10], visual_refs[:10]  # limit to top 10 each to avoid massive outputs

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
            content, thermal, interpretation, neg, pos = "Not Available", "Not Available", "Not Available", [], []
            try:
                prompt = AREA_ANALYSIS_PROMPT.format(
                    area=area,
                    observations_text=obs_text,
                )
                analysis_json = call_llm_json(prompt, system_prompt=DDR_SYNTHESIS_SYSTEM)
                if isinstance(analysis_json, dict):
                    content = analysis_json.get("content", "Not Available")
                    thermal = analysis_json.get("thermal_findings", "Not Available")
                    interpretation = analysis_json.get("combined_interpretation", "Not Available")
                    neg = analysis_json.get("negative_inputs", [])
                    pos = analysis_json.get("positive_inputs", [])
            except Exception as e:
                logger.error(f"Failed area analysis for '{area}': {e}")
                content = obs_text

            # Map images to this area
            area_images = self._map_images_to_area(area, area_obs, images)

            section = DDRSection(
                title=area,
                content=content,
                thermal_findings=thermal,
                combined_interpretation=interpretation,
                negative_inputs=neg,
                positive_inputs=pos,
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
        except (ImportError, OSError, Exception) as e:
            logger.warning(
                f"WeasyPrint dependency missing or failed to load. Skipping PDF export. "
                f"Error: {e}\n"
                f"(On Windows, you need GTK3 installed for PDF export)"
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
                "completeness": "Data Completeness",
            }
            for key, label in labels.items():
                if key in report.metadata:
                    lines.append(f"| {label} | {report.metadata[key]} |")
            lines.append("")

        # 1. Introduction
        lines.append("\n## 1. Introduction\n")
        lines.append(f"**Background of Inspection:** {report.introduction.background}")
        lines.append(f"**Objective of Assessment:** {report.introduction.objective}")
        lines.append(f"**Scope of Work:** {report.introduction.scope}")
        lines.append(f"**Tools Used:** {report.introduction.tools_used}")
        lines.append("")

        # 2. General Information
        lines.append("\n## 2. General Information\n")
        lines.append(f"- **Client Details:** {report.general_info.client_details}")
        lines.append(f"- **Location:** {report.general_info.location}")
        lines.append(f"- **Structure Type:** {report.general_info.structure_type}")
        lines.append(f"- **Age of Building:** {report.general_info.age_of_building}")
        lines.append(f"- **Floors:** {report.general_info.floors}")
        lines.append(f"- **Previous Repairs:** {report.general_info.previous_repairs}")
        lines.append("")

        # 3. Area Observations
        lines.append("\n## 3. Visual Observations and Readings\n")
        for idx, section in enumerate(report.area_observations, 1):
            lines.append(f"\n### 3.{idx} {section.title}\n")
            
            lines.append("**A. Observations**")
            lines.append(section.content + "\n")
            
            lines.append("**B. Supporting Thermal Findings**")
            lines.append(section.thermal_findings + "\n")
            
            lines.append("**C. Combined Interpretation**")
            lines.append(section.combined_interpretation + "\n")
            
            lines.append("**Negative Side Inputs (Issues Observed):**")
            for item in section.negative_inputs:
                lines.append(f"- {item}")
            lines.append("")
            
            lines.append("**Positive Side Inputs (Source of Issue):**")
            for item in section.positive_inputs:
                lines.append(f"- {item}")
            lines.append("")

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

        # 4. Analysis and Suggestions
        lines.append("\n## 4. Analysis & Suggestions\n")
        lines.append(f"### 4.1 Probable Root Cause\n{report.root_causes}\n")
        lines.append(f"### 4.2 Severity Assessment\n{report.severity_assessment}\n")
        lines.append(f"### 4.3 Recommended Actions\n{report.recommended_actions}\n")
        
        lines.append("### 4.4 Summary Table\n")
        if report.summary_table:
            lines.append("| Issue | Affected Area | Cause | Severity | Recommended Action |")
            lines.append("|-------|---------------|-------|----------|--------------------|")
            for row in report.summary_table:
                lines.append(f"| {row.get('issue','')} | {row.get('affected_area','')} | {row.get('cause','')} | {row.get('severity','')} | {row.get('recommended_action','')} |")
        lines.append("")

        # 5. Image References
        lines.append("\n## 5. Image References\n")
        lines.append("### Thermal References\n")
        for i, img in enumerate(report.thermal_image_refs, 1):
            lines.append(f"**IMAGE {i}: {img.get('description','')}** (Page {img.get('page','')})")
        lines.append("\n### Visual References\n")
        for i, img in enumerate(report.visual_image_refs, 1):
            lines.append(f"**IMAGE {i}: {img.get('description','')}** (Page {img.get('page','')})")

        # 6. Additional Notes
        lines.append("\n\n## 6. Additional Notes\n")
        lines.append(report.additional_notes)

        # 7. Missing Info
        lines.append("\n\n## 7. Missing or Unclear Information\n")
        lines.append(report.missing_info)

        # 8. Limitations & Disclaimer
        lines.append("\n\n## 8. Limitations and Disclaimer\n")
        lines.append("- This inspection is visual and non-exhaustive.\n- Hidden defects may not be detected.\n- Recommend expert consultation for critical issues.")

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

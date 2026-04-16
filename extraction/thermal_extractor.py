"""
Thermal Report Extractor Module.

Specialized LLM-based extractor for thermal imaging / infrared survey reports.
Handles thermal-specific terminology like temperature differentials, hot spots,
moisture intrusion indicators, and thermal anomaly classifications.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from loguru import logger

from parser.pdf_parser import ParsedDocument
from llm_client import call_llm_json


@dataclass
class ThermalObservation:
    """A structured observation from a thermal/infrared report."""

    area: str
    observation: str
    severity_hint: str = ""
    recommendation_hint: str = ""
    source: str = ""
    page: str = ""
    confidence: float = 0.0
    temperature_data: str = ""  # e.g., "Delta-T: 8.2°F"
    thermal_pattern: str = ""  # e.g., "moisture intrusion", "insulation void"
    image_reference: str = ""  # e.g., "IR Image 3-A"

    def to_dict(self) -> dict:
        return asdict(self)


THERMAL_SYSTEM_PROMPT = """You are an expert thermographer and building diagnostics specialist.
Your task is to extract structured observations from thermal/infrared survey reports.

RULES:
- Extract EVERY thermal anomaly, finding, or observation mentioned.
- Capture temperature data (delta-T, surface temps) when provided.
- Identify thermal patterns: moisture intrusion, insulation voids/gaps,
  air leakage, electrical hotspots, HVAC issues, etc.
- Note any IR image references mentioned (e.g., "Image 3", "Figure 2-A").
- Preserve exact terminology and measurements from the report.
- Set area to the specific location (e.g., "Master Bedroom - North Wall").
- Map severity hints:
    - critical: safety risk, active moisture damage, electrical hazard
    - high: significant heat loss, active leaks, major insulation failure
    - medium: moderate anomalies, partial insulation gaps
    - low: minor temperature variance, cosmetic issues
    - informational: normal readings noted for baseline
- NEVER invent data not present in the text.
"""

THERMAL_EXTRACTION_PROMPT = """Analyze the following thermal/infrared report text and extract ALL observations.

SOURCE DOCUMENT: {source_name}
PAGE RANGE: {page_range}

--- BEGIN TEXT ---
{text}
--- END TEXT ---

Return a JSON array of observation objects. Each object must have:
{{
  "area": "<specific location or component>",
  "observation": "<thermal finding — include temperature data if available>",
  "severity_hint": "<critical|high|medium|low|informational or empty>",
  "recommendation_hint": "<suggested action if mentioned>",
  "source": "{source_name}",
  "page": "<page number(s)>",
  "temperature_data": "<any temperature readings or delta-T values>",
  "thermal_pattern": "<moisture|insulation|air_leakage|electrical|hvac|structural|other or empty>",
  "image_reference": "<referenced IR image identifier if any>"
}}

Return ONLY the JSON array. No other text.
"""


class ThermalExtractor:
    """
    Specialized extractor for thermal imaging reports.

    Extends the observation extraction pattern with thermal-specific
    fields like temperature data, thermal patterns, and IR image references.
    """

    def __init__(self, chunk_pages: int = 5):
        """
        Initialize the thermal extractor.

        Args:
            chunk_pages: Number of pages to process per LLM call.
        """
        self.chunk_pages = chunk_pages

    def extract(self, parsed_doc: ParsedDocument) -> list[ThermalObservation]:
        """
        Extract thermal observations from a parsed thermal report.

        Args:
            parsed_doc: A ParsedDocument from the PDF parser.

        Returns:
            List of ThermalObservation objects.
        """
        logger.info(
            f"Extracting thermal observations from '{parsed_doc.file_name}' "
            f"({parsed_doc.total_pages} pages)"
        )

        all_observations: list[ThermalObservation] = []
        chunks = self._create_chunks(parsed_doc)

        for chunk_idx, (text, page_range) in enumerate(chunks):
            if not text.strip():
                continue

            logger.debug(
                f"Processing thermal chunk {chunk_idx + 1}/{len(chunks)} "
                f"(pages {page_range})"
            )

            try:
                obs_list = self._extract_chunk(
                    text=text,
                    source_name=parsed_doc.file_name,
                    page_range=page_range,
                )
                all_observations.extend(obs_list)
            except Exception as e:
                logger.error(
                    f"Failed thermal extraction for chunk {chunk_idx + 1} "
                    f"(pages {page_range}): {e}"
                )

        deduped = self._deduplicate(all_observations)

        logger.info(
            f"Extracted {len(deduped)} thermal observations from "
            f"'{parsed_doc.file_name}' (before dedup: {len(all_observations)})"
        )
        return deduped

    def _create_chunks(
        self, parsed_doc: ParsedDocument
    ) -> list[tuple[str, str]]:
        """Split document into text chunks with page ranges."""
        chunks = []
        pages = parsed_doc.pages

        for i in range(0, len(pages), self.chunk_pages):
            batch = pages[i : i + self.chunk_pages]
            text_parts = []
            for p in batch:
                if p.raw_text.strip():
                    text_parts.append(
                        f"[Page {p.page_number}]\n{p.raw_text}"
                    )

            if text_parts:
                combined_text = "\n\n".join(text_parts)
                page_range = f"{batch[0].page_number}-{batch[-1].page_number}"
                chunks.append((combined_text, page_range))

        return chunks

    def _extract_chunk(
        self,
        text: str,
        source_name: str,
        page_range: str,
    ) -> list[ThermalObservation]:
        """Send text chunk to LLM and parse thermal observations."""
        prompt = THERMAL_EXTRACTION_PROMPT.format(
            source_name=source_name,
            page_range=page_range,
            text=text[:12000],
        )

        raw_observations = call_llm_json(
            prompt, system_prompt=THERMAL_SYSTEM_PROMPT
        )

        if not isinstance(raw_observations, list):
            raw_observations = [raw_observations]

        observations = []
        for raw in raw_observations:
            if not isinstance(raw, dict):
                continue

            obs = ThermalObservation(
                area=raw.get("area", "").strip(),
                observation=raw.get("observation", "").strip(),
                severity_hint=raw.get("severity_hint", "").strip().lower(),
                recommendation_hint=raw.get("recommendation_hint", "").strip(),
                source=raw.get("source", source_name).strip(),
                page=str(raw.get("page", "")).strip(),
                confidence=float(raw.get("confidence", 0.85)),
                temperature_data=raw.get("temperature_data", "").strip(),
                thermal_pattern=raw.get("thermal_pattern", "").strip().lower(),
                image_reference=raw.get("image_reference", "").strip(),
            )
            if obs.area and obs.observation:
                observations.append(obs)

        return observations

    def _deduplicate(
        self, observations: list[ThermalObservation]
    ) -> list[ThermalObservation]:
        """Remove near-exact duplicate thermal observations."""
        seen = set()
        unique = []

        for obs in observations:
            key = (
                obs.area.lower().strip(),
                obs.observation.lower().strip()[:100],
            )
            if key not in seen:
                seen.add(key)
                unique.append(obs)

        return unique


def extract_thermal_observations(
    parsed_doc: ParsedDocument,
) -> list[ThermalObservation]:
    """
    Convenience function for one-shot thermal extraction.

    Args:
        parsed_doc: Parsed thermal report document.

    Returns:
        List of structured ThermalObservation objects.
    """
    extractor = ThermalExtractor()
    return extractor.extract(parsed_doc)

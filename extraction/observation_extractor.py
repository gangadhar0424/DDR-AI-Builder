"""
Observation Extractor Module.

Uses LLM prompts to convert raw text from inspection reports into
structured observation JSON objects with area, severity, and
recommendation hints. Maintains source traceability via page numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from loguru import logger

from parser.pdf_parser import ParsedDocument
from llm_client import call_llm_json


@dataclass
class Observation:
    """A single structured observation extracted from a report."""

    area: str
    observation: str
    severity_hint: str = ""
    recommendation_hint: str = ""
    source: str = ""
    page: str = ""
    confidence: float = 0.0
    raw_text_snippet: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


SYSTEM_PROMPT = """You are an expert building inspector and report analyst.
Your task is to extract structured observations from inspection report text.

RULES:
- Extract EVERY distinct observation, defect, issue, or finding mentioned.
- Each observation must be specific and actionable — not vague summaries.
- Preserve the exact terminology used in the report.
- If severity is mentioned or implied, capture it in severity_hint.
- If a recommendation or action is mentioned, capture it in recommendation_hint.
- Set area to the specific location/component (e.g., "Roof - North Slope", "Bathroom 2 - Shower").
- NEVER invent or assume information not present in the text.
- If a field cannot be determined, set it to an empty string.
"""

EXTRACTION_PROMPT = """Analyze the following inspection report text and extract ALL observations.

SOURCE DOCUMENT: {source_name}
PAGE RANGE: {page_range}

--- BEGIN TEXT ---
{text}
--- END TEXT ---

Return a JSON array of observation objects. Each object must have these fields:
{{
  "area": "<specific location or component>",
  "observation": "<what was found — be precise>",
  "severity_hint": "<critical|high|medium|low|informational or empty>",
  "recommendation_hint": "<suggested action if mentioned, else empty>",
  "source": "{source_name}",
  "page": "<page number(s) where this was found>"
}}

Return ONLY the JSON array. Do not include any other text.
"""


class ObservationExtractor:
    """
    Extracts structured observations from parsed inspection report PDFs.

    Processes the document in page-level chunks to stay within LLM context
    limits while preserving page-level source traceability.
    """

    def __init__(self, chunk_pages: int = 5):
        """
        Initialize the extractor.

        Args:
            chunk_pages: Number of pages to process per LLM call.
                Larger chunks give more context but cost more tokens.
        """
        self.chunk_pages = chunk_pages

    def extract(self, parsed_doc: ParsedDocument) -> list[Observation]:
        """
        Extract observations from a parsed inspection document.

        Args:
            parsed_doc: A ParsedDocument from the PDF parser.

        Returns:
            List of Observation objects with source traceability.
        """
        logger.info(
            f"Extracting observations from '{parsed_doc.file_name}' "
            f"({parsed_doc.total_pages} pages)"
        )

        all_observations: list[Observation] = []

        # Process in chunks to manage LLM token limits
        chunks = self._create_chunks(parsed_doc)

        for chunk_idx, (text, page_range) in enumerate(chunks):
            if not text.strip():
                continue

            logger.debug(
                f"Processing chunk {chunk_idx + 1}/{len(chunks)} "
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
                    f"Failed to extract observations from chunk {chunk_idx + 1} "
                    f"(pages {page_range}): {e}"
                )

        # Deduplicate within the same document
        deduped = self._deduplicate(all_observations)

        logger.info(
            f"Extracted {len(deduped)} observations from '{parsed_doc.file_name}' "
            f"(before dedup: {len(all_observations)})"
        )
        return deduped

    def _create_chunks(
        self, parsed_doc: ParsedDocument
    ) -> list[tuple[str, str]]:
        """
        Split parsed document into text chunks with page ranges.

        Returns:
            List of (text, page_range_string) tuples.
        """
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
    ) -> list[Observation]:
        """
        Send a text chunk to LLM and parse the resulting observations.

        Args:
            text: Combined page text for this chunk.
            source_name: Name of the source PDF.
            page_range: Page range string (e.g., "1-5").

        Returns:
            List of Observation objects parsed from LLM response.
        """
        prompt = EXTRACTION_PROMPT.format(
            source_name=source_name,
            page_range=page_range,
            text=text[:12000],  # Cap text to avoid token overflow
        )

        raw_observations = call_llm_json(prompt, system_prompt=SYSTEM_PROMPT)

        if not isinstance(raw_observations, list):
            raw_observations = [raw_observations]

        observations = []
        for raw in raw_observations:
            if not isinstance(raw, dict):
                continue
            obs = Observation(
                area=raw.get("area", "").strip(),
                observation=raw.get("observation", "").strip(),
                severity_hint=raw.get("severity_hint", "").strip().lower(),
                recommendation_hint=raw.get("recommendation_hint", "").strip(),
                source=raw.get("source", source_name).strip(),
                page=str(raw.get("page", "")).strip(),
                confidence=float(raw.get("confidence", 0.85)),
            )
            # Only keep observations with meaningful content
            if obs.area and obs.observation:
                observations.append(obs)

        return observations

    def _deduplicate(
        self, observations: list[Observation]
    ) -> list[Observation]:
        """
        Remove near-exact duplicate observations within the same source.

        Uses normalized text comparison to catch trivial duplicates.
        Semantic deduplication across sources is handled by the merger.
        """
        seen = set()
        unique = []

        for obs in observations:
            # Normalize for comparison
            key = (
                obs.area.lower().strip(),
                obs.observation.lower().strip()[:100],
            )
            if key not in seen:
                seen.add(key)
                unique.append(obs)

        return unique


def extract_observations(parsed_doc: ParsedDocument) -> list[Observation]:
    """
    Convenience function for one-shot observation extraction.

    Args:
        parsed_doc: Parsed inspection report document.

    Returns:
        List of structured Observation objects.
    """
    extractor = ObservationExtractor()
    return extractor.extract(parsed_doc)

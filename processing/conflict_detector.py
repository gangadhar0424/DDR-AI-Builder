"""
Conflict Detector Module.

Identifies conflicting statements between inspection and thermal
observations. Two observations are considered conflicting when they
reference the same area/component but contain contradictory assessments,
severity ratings, or factual claims.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from loguru import logger

from llm_client import call_llm_json
from processing.merger import MergedObservation


@dataclass
class Conflict:
    """A detected conflict between observations."""

    area: str
    observation_1: str
    source_1: str
    observation_2: str
    source_2: str
    conflict_type: str  # "severity", "factual", "assessment", "recommendation"
    description: str
    resolution_suggestion: str = ""
    severity_impact: str = ""  # How this conflict affects overall assessment


@dataclass
class ConflictReport:
    """Complete conflict detection results."""

    conflicts: list[Conflict] = field(default_factory=list)
    total_checked: int = 0
    total_conflicts: int = 0

    def has_conflicts(self) -> bool:
        return len(self.conflicts) > 0

    def to_dict(self) -> dict:
        return {
            "total_checked": self.total_checked,
            "total_conflicts": self.total_conflicts,
            "conflicts": [asdict(c) for c in self.conflicts],
        }


CONFLICT_SYSTEM_PROMPT = """You are a building diagnostics expert reviewing observations 
from two different reports (inspection and thermal) about the same property.

Your task is to identify GENUINE CONFLICTS — cases where the two sources
directly contradict each other. 

A conflict exists when:
- One report says an area is in good condition, the other reports damage
- Severity assessments directly contradict (e.g., "minor" vs "critical")
- Factual claims conflict (e.g., "no moisture" vs "active moisture detected")
- Recommendations contradict (e.g., "monitor only" vs "immediate repair needed")

Do NOT flag as conflicts:
- Complementary information (one report adds detail the other doesn't have)
- Different observations about different aspects of the same area
- One report being more detailed than the other
"""

CONFLICT_DETECTION_PROMPT = """Review the following corroborated observations (found in both reports)
and identify any GENUINE CONFLICTS between the inspection and thermal findings.

OBSERVATIONS:
{observations_text}

For each genuine conflict found, return a JSON array of conflict objects:
{{
  "area": "<area where conflict exists>",
  "observation_1": "<first conflicting statement>",
  "source_1": "<source of first statement>",
  "observation_2": "<second conflicting statement>",
  "source_2": "<source of second statement>",
  "conflict_type": "<severity|factual|assessment|recommendation>",
  "description": "<clear explanation of the contradiction>",
  "resolution_suggestion": "<how to resolve this conflict>",
  "severity_impact": "<how this affects overall assessment>"
}}

If NO genuine conflicts exist, return an empty JSON array: []

Return ONLY the JSON array.
"""


class ConflictDetector:
    """
    Detects conflicting statements between merged observations.

    Focuses on corroborated observations (those found in both reports)
    since conflicts between independent observations are less meaningful.
    Uses LLM reasoning to identify genuine contradictions vs. complementary
    information.
    """

    def detect(
        self,
        merged_observations: list[MergedObservation],
    ) -> ConflictReport:
        """
        Analyze merged observations for conflicts.

        Args:
            merged_observations: List of MergedObservation from the merger.

        Returns:
            ConflictReport with all detected conflicts.
        """
        # Focus on corroborated observations (found in both reports)
        corroborated = [
            obs for obs in merged_observations if obs.is_corroborated
        ]

        report = ConflictReport(total_checked=len(corroborated))

        if not corroborated:
            logger.info("No corroborated observations to check for conflicts")
            return report

        logger.info(f"Checking {len(corroborated)} corroborated observations for conflicts")

        # Process in batches to stay within token limits
        batch_size = 10
        for i in range(0, len(corroborated), batch_size):
            batch = corroborated[i : i + batch_size]
            try:
                conflicts = self._detect_batch(batch)
                report.conflicts.extend(conflicts)
            except Exception as e:
                logger.error(f"Conflict detection failed for batch {i // batch_size + 1}: {e}")

        report.total_conflicts = len(report.conflicts)

        if report.has_conflicts():
            logger.warning(
                f"Detected {report.total_conflicts} conflicts in observations"
            )
        else:
            logger.info("No conflicts detected between reports")

        return report

    def _detect_batch(
        self, observations: list[MergedObservation]
    ) -> list[Conflict]:
        """Run conflict detection on a batch of observations."""
        # Format observations for the prompt
        obs_text_parts = []
        for idx, obs in enumerate(observations, 1):
            sources_str = ", ".join(obs.sources)
            obs_text_parts.append(
                f"[{idx}] Area: {obs.area}\n"
                f"    Observation: {obs.observation}\n"
                f"    Severity: {obs.severity}\n"
                f"    Sources: {sources_str}\n"
                f"    Recommendation: {obs.recommendation}"
            )

        observations_text = "\n\n".join(obs_text_parts)

        prompt = CONFLICT_DETECTION_PROMPT.format(
            observations_text=observations_text
        )

        raw_conflicts = call_llm_json(prompt, system_prompt=CONFLICT_SYSTEM_PROMPT)

        if not isinstance(raw_conflicts, list):
            raw_conflicts = [raw_conflicts]

        conflicts = []
        for raw in raw_conflicts:
            if not isinstance(raw, dict):
                continue
            # Only include if there's a real conflict description
            if not raw.get("description"):
                continue

            conflict = Conflict(
                area=raw.get("area", ""),
                observation_1=raw.get("observation_1", ""),
                source_1=raw.get("source_1", ""),
                observation_2=raw.get("observation_2", ""),
                source_2=raw.get("source_2", ""),
                conflict_type=raw.get("conflict_type", "assessment"),
                description=raw.get("description", ""),
                resolution_suggestion=raw.get("resolution_suggestion", ""),
                severity_impact=raw.get("severity_impact", ""),
            )
            conflicts.append(conflict)

        return conflicts


def detect_conflicts(
    merged_observations: list[MergedObservation],
) -> ConflictReport:
    """
    Convenience function for conflict detection.

    Args:
        merged_observations: Merged observations from the merger.

    Returns:
        ConflictReport with detected conflicts.
    """
    detector = ConflictDetector()
    return detector.detect(merged_observations)

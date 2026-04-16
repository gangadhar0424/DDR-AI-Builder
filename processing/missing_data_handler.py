"""
Missing Data Handler Module.

Scans merged observations for missing, incomplete, or unclear information.
Marks absent fields as "Not Available" and generates a structured report
of data gaps that should be highlighted in the final DDR.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from loguru import logger

from processing.merger import MergedObservation, MergeResult


@dataclass
class MissingField:
    """A single missing or incomplete data field."""

    area: str
    field_name: str
    status: str  # "missing", "unclear", "incomplete"
    description: str
    impact: str = ""  # How this gap affects the assessment


@dataclass
class MissingDataReport:
    """Complete report of all missing or unclear data."""

    missing_fields: list[MissingField] = field(default_factory=list)
    total_observations_checked: int = 0
    observations_with_gaps: int = 0
    completeness_score: float = 0.0  # 0.0 to 1.0

    def to_dict(self) -> dict:
        return {
            "total_observations_checked": self.total_observations_checked,
            "observations_with_gaps": self.observations_with_gaps,
            "completeness_score": round(self.completeness_score, 3),
            "missing_fields": [asdict(f) for f in self.missing_fields],
        }

    @property
    def summary(self) -> str:
        """Human-readable summary of missing data."""
        if not self.missing_fields:
            return "All observations are complete — no missing data detected."
        return (
            f"{len(self.missing_fields)} data gaps found across "
            f"{self.observations_with_gaps} observations "
            f"(completeness: {self.completeness_score:.0%})"
        )


# Fields that every observation should ideally have
REQUIRED_FIELDS = {
    "area": "Location / Area",
    "observation": "Observation Details",
    "severity": "Severity Level",
    "recommendation": "Recommended Action",
}

OPTIONAL_FIELDS = {
    "temperature_data": "Temperature Data",
    "thermal_pattern": "Thermal Pattern Classification",
    "sources": "Source Documents",
    "pages": "Page References",
}


class MissingDataHandler:
    """
    Scans observations for completeness and fills default values.

    Checks each observation against required and optional field lists,
    flags missing data, and fills absent values with "Not Available"
    to ensure the final DDR doesn't contain empty fields.
    """

    def __init__(self, fill_defaults: bool = True):
        """
        Initialize the handler.

        Args:
            fill_defaults: If True, fill missing fields with "Not Available".
        """
        self.fill_defaults = fill_defaults

    def process(
        self,
        merge_result: MergeResult,
    ) -> tuple[MergeResult, MissingDataReport]:
        """
        Scan and patch merged observations for missing data.

        Args:
            merge_result: MergeResult from the observation merger.

        Returns:
            Tuple of (patched MergeResult, MissingDataReport).
        """
        logger.info(
            f"Checking {len(merge_result.merged_observations)} "
            f"observations for missing data"
        )

        report = MissingDataReport(
            total_observations_checked=len(merge_result.merged_observations),
        )

        total_fields = 0
        complete_fields = 0
        obs_with_gaps = set()

        for idx, obs in enumerate(merge_result.merged_observations):
            gaps = self._check_observation(obs, idx)
            if gaps:
                obs_with_gaps.add(idx)
                report.missing_fields.extend(gaps)

            # Count fields for completeness score
            field_count, filled_count = self._count_fields(obs)
            total_fields += field_count
            complete_fields += filled_count

            # Fill defaults if enabled
            if self.fill_defaults:
                self._fill_defaults(obs)

        report.observations_with_gaps = len(obs_with_gaps)
        report.completeness_score = (
            complete_fields / total_fields if total_fields > 0 else 1.0
        )

        logger.info(
            f"Data completeness: {report.completeness_score:.1%} — "
            f"{len(report.missing_fields)} gaps in "
            f"{report.observations_with_gaps} observations"
        )

        return merge_result, report

    def _check_observation(
        self,
        obs: MergedObservation,
        index: int,
    ) -> list[MissingField]:
        """Check a single observation for missing required fields."""
        gaps = []

        # Check required fields
        for field_key, field_label in REQUIRED_FIELDS.items():
            value = getattr(obs, field_key, None)
            if not value or (isinstance(value, str) and not value.strip()):
                gaps.append(
                    MissingField(
                        area=obs.area or f"Observation #{index + 1}",
                        field_name=field_label,
                        status="missing",
                        description=f"{field_label} is not provided.",
                        impact=self._assess_impact(field_key),
                    )
                )
            elif isinstance(value, str) and self._is_unclear(value):
                gaps.append(
                    MissingField(
                        area=obs.area,
                        field_name=field_label,
                        status="unclear",
                        description=f"{field_label} contains vague or unclear content: '{value[:80]}'",
                        impact="May require clarification for accurate assessment.",
                    )
                )

        # Check source traceability
        if not obs.sources:
            gaps.append(
                MissingField(
                    area=obs.area or f"Observation #{index + 1}",
                    field_name="Source Documents",
                    status="missing",
                    description="No source document reference provided.",
                    impact="Cannot verify the origin of this observation.",
                )
            )

        if not obs.pages:
            gaps.append(
                MissingField(
                    area=obs.area or f"Observation #{index + 1}",
                    field_name="Page References",
                    status="missing",
                    description="No page number reference provided.",
                    impact="Cannot trace back to the original report page.",
                )
            )

        # Check thermal-specific fields for thermal-sourced observations
        if any("thermal" in s.lower() for s in obs.sources):
            if not obs.temperature_data:
                gaps.append(
                    MissingField(
                        area=obs.area or f"Observation #{index + 1}",
                        field_name="Temperature Data",
                        status="missing",
                        description="Thermal observation lacks temperature readings.",
                        impact="Quantitative severity assessment may be limited.",
                    )
                )

        return gaps

    def _count_fields(self, obs: MergedObservation) -> tuple[int, int]:
        """Count total and filled fields for an observation."""
        total = 0
        filled = 0

        for field_key in REQUIRED_FIELDS:
            total += 1
            value = getattr(obs, field_key, None)
            if value and (not isinstance(value, str) or value.strip()):
                filled += 1

        return total, filled

    def _fill_defaults(self, obs: MergedObservation) -> None:
        """Fill missing string fields with 'Not Available'."""
        if not obs.severity or not obs.severity.strip():
            obs.severity = "Not Available"

        if not obs.recommendation or not obs.recommendation.strip():
            obs.recommendation = "Not Available"

        if not obs.temperature_data:
            obs.temperature_data = ""

        if not obs.thermal_pattern:
            obs.thermal_pattern = ""

    def _is_unclear(self, text: str) -> bool:
        """Check if text content is vague or unclear."""
        unclear_indicators = [
            "unclear", "unknown", "not sure", "possibly",
            "maybe", "tbd", "to be determined", "n/a",
            "see above", "refer to", "as discussed",
        ]
        text_lower = text.lower().strip()
        return any(indicator in text_lower for indicator in unclear_indicators)

    def _assess_impact(self, field_key: str) -> str:
        """Assess the impact of a missing field."""
        impact_map = {
            "area": "Cannot determine the location of the issue.",
            "observation": "Core finding is missing — observation is unusable.",
            "severity": "Cannot prioritize this issue for remediation.",
            "recommendation": "No actionable guidance can be provided.",
        }
        return impact_map.get(field_key, "May affect report completeness.")


def handle_missing_data(
    merge_result: MergeResult,
) -> tuple[MergeResult, MissingDataReport]:
    """
    Convenience function for missing data handling.

    Args:
        merge_result: MergeResult from the observation merger.

    Returns:
        Tuple of (patched MergeResult, MissingDataReport).
    """
    handler = MissingDataHandler()
    return handler.process(merge_result)

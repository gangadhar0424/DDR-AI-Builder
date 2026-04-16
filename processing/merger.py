"""
Observation Merger Module.

Merges observations from inspection and thermal reports using:
1. Semantic similarity via SentenceTransformers
2. Area/location matching
3. Keyword overlap scoring

Produces deduplicated, merged observations grouped by area with
confidence scores and source traceability.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from loguru import logger

import config

# Lazy-load SentenceTransformers to avoid slow startup
_embedding_model = None


def _get_embedding_model():
    """Lazy-load the sentence transformer model."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")
        _embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _embedding_model


@dataclass
class MergedObservation:
    """A merged observation combining data from multiple sources."""

    area: str
    observation: str
    severity: str = ""
    recommendation: str = ""
    sources: list[str] = field(default_factory=list)
    pages: list[str] = field(default_factory=list)
    confidence_score: float = 0.0
    merge_method: str = ""  # "semantic", "area_match", "keyword"
    temperature_data: str = ""
    thermal_pattern: str = ""
    image_references: list[str] = field(default_factory=list)
    is_corroborated: bool = False  # Found in both reports

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MergeResult:
    """Complete result of the observation merging process."""

    merged_observations: list[MergedObservation] = field(default_factory=list)
    total_inspection_obs: int = 0
    total_thermal_obs: int = 0
    total_merged: int = 0
    corroborated_count: int = 0
    areas: list[str] = field(default_factory=list)

    def get_by_area(self, area: str) -> list[MergedObservation]:
        """Get all observations for a specific area."""
        area_lower = area.lower()
        return [
            obs for obs in self.merged_observations
            if area_lower in obs.area.lower()
        ]

    def get_by_severity(self, severity: str) -> list[MergedObservation]:
        """Get all observations of a specific severity level."""
        sev_lower = severity.lower()
        return [
            obs for obs in self.merged_observations
            if obs.severity.lower() == sev_lower
        ]


class ObservationMerger:
    """
    Merges observations from inspection and thermal reports.

    Uses a multi-strategy approach:
    1. Compute semantic embeddings for all observations
    2. Find matches above similarity threshold
    3. Merge matched pairs into single observations
    4. Keep unmatched observations from both sources
    5. Assign confidence scores based on corroboration
    """

    def __init__(
        self,
        similarity_threshold: float | None = None,
    ):
        """
        Initialize the merger.

        Args:
            similarity_threshold: Minimum cosine similarity to consider
                two observations as referring to the same issue.
        """
        self.similarity_threshold = (
            similarity_threshold or config.SIMILARITY_THRESHOLD
        )

    def merge(
        self,
        inspection_obs: list,
        thermal_obs: list,
    ) -> MergeResult:
        """
        Merge observations from both inspection and thermal reports.

        Args:
            inspection_obs: List of Observation objects from inspection report.
            thermal_obs: List of ThermalObservation objects from thermal report.

        Returns:
            MergeResult with deduplicated, merged observations.
        """
        logger.info(
            f"Merging {len(inspection_obs)} inspection + "
            f"{len(thermal_obs)} thermal observations"
        )

        result = MergeResult(
            total_inspection_obs=len(inspection_obs),
            total_thermal_obs=len(thermal_obs),
        )

        if not inspection_obs and not thermal_obs:
            logger.warning("No observations to merge")
            return result

        # Convert all observations to a common format
        insp_items = [self._normalize(obs, "inspection") for obs in inspection_obs]
        therm_items = [self._normalize(obs, "thermal") for obs in thermal_obs]

        # Find matches using semantic similarity
        matches, unmatched_insp, unmatched_therm = self._find_matches(
            insp_items, therm_items
        )

        # Create merged observations from matches
        for insp_idx, therm_idx, similarity in matches:
            merged = self._merge_pair(
                insp_items[insp_idx],
                therm_items[therm_idx],
                similarity,
            )
            result.merged_observations.append(merged)
            result.corroborated_count += 1

        # Add unmatched inspection observations
        for idx in unmatched_insp:
            merged = self._single_to_merged(insp_items[idx])
            result.merged_observations.append(merged)

        # Add unmatched thermal observations
        for idx in unmatched_therm:
            merged = self._single_to_merged(therm_items[idx])
            result.merged_observations.append(merged)

        # Sort by severity priority then area
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "informational": 4, "": 5}
        result.merged_observations.sort(
            key=lambda o: (
                severity_order.get(o.severity.lower(), 5),
                o.area.lower(),
            )
        )

        # Collect unique areas
        areas = set()
        for obs in result.merged_observations:
            if obs.area:
                # Normalize area to top-level location
                top_area = obs.area.split(" - ")[0].strip() if " - " in obs.area else obs.area
                areas.add(top_area)
        result.areas = sorted(areas)
        result.total_merged = len(result.merged_observations)

        logger.info(
            f"Merge complete: {result.total_merged} observations, "
            f"{result.corroborated_count} corroborated, "
            f"{len(result.areas)} areas"
        )
        return result

    def _normalize(self, obs, source_type: str) -> dict:
        """Convert an observation object to a normalized dictionary."""
        if hasattr(obs, "to_dict"):
            d = obs.to_dict()
        elif isinstance(obs, dict):
            d = obs.copy()
        else:
            d = {"area": str(obs), "observation": str(obs)}

        d["_source_type"] = source_type
        d.setdefault("area", "")
        d.setdefault("observation", "")
        d.setdefault("severity_hint", "")
        d.setdefault("recommendation_hint", "")
        d.setdefault("source", "")
        d.setdefault("page", "")
        d.setdefault("temperature_data", "")
        d.setdefault("thermal_pattern", "")
        d.setdefault("image_reference", "")

        # Create a text representation for embedding
        d["_embed_text"] = f"{d['area']}. {d['observation']}"
        return d

    def _find_matches(
        self,
        insp_items: list[dict],
        therm_items: list[dict],
    ) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
        """
        Find matching observations between inspection and thermal lists.

        Returns:
            Tuple of (matches, unmatched_insp_indices, unmatched_therm_indices).
            Each match is (insp_idx, therm_idx, similarity_score).
        """
        if not insp_items or not therm_items:
            return (
                [],
                list(range(len(insp_items))),
                list(range(len(therm_items))),
            )

        # Compute embeddings
        model = _get_embedding_model()
        insp_texts = [item["_embed_text"] for item in insp_items]
        therm_texts = [item["_embed_text"] for item in therm_items]

        logger.debug("Computing embeddings for similarity matching...")
        insp_embeddings = model.encode(insp_texts, normalize_embeddings=True)
        therm_embeddings = model.encode(therm_texts, normalize_embeddings=True)

        # Compute cosine similarity matrix
        import numpy as np

        sim_matrix = np.dot(insp_embeddings, therm_embeddings.T)

        # Greedy matching: find best pairs above threshold
        matches = []
        used_insp = set()
        used_therm = set()

        # Sort all similarities in descending order
        pairs = []
        for i in range(len(insp_items)):
            for j in range(len(therm_items)):
                if sim_matrix[i][j] >= self.similarity_threshold:
                    pairs.append((sim_matrix[i][j], i, j))

        pairs.sort(reverse=True)

        for sim, i, j in pairs:
            if i not in used_insp and j not in used_therm:
                # Additional area-matching boost
                area_sim = self._area_similarity(
                    insp_items[i]["area"], therm_items[j]["area"]
                )
                if area_sim > 0.3 or sim >= self.similarity_threshold + 0.05:
                    matches.append((i, j, float(sim)))
                    used_insp.add(i)
                    used_therm.add(j)

        unmatched_insp = [i for i in range(len(insp_items)) if i not in used_insp]
        unmatched_therm = [j for j in range(len(therm_items)) if j not in used_therm]

        logger.debug(
            f"Found {len(matches)} matches, "
            f"{len(unmatched_insp)} unmatched inspection, "
            f"{len(unmatched_therm)} unmatched thermal"
        )
        return matches, unmatched_insp, unmatched_therm

    def _area_similarity(self, area1: str, area2: str) -> float:
        """
        Compute simple area similarity based on word overlap.

        Returns a value between 0.0 and 1.0.
        """
        if not area1 or not area2:
            return 0.0

        words1 = set(area1.lower().split())
        words2 = set(area2.lower().split())

        # Remove common stop words
        stop_words = {"the", "a", "an", "of", "in", "at", "on", "to", "-", "–"}
        words1 -= stop_words
        words2 -= stop_words

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / len(union)

    def _merge_pair(
        self,
        insp: dict,
        therm: dict,
        similarity: float,
    ) -> MergedObservation:
        """Merge a matched pair of observations into one."""
        # Combine observations, preferring the more detailed one
        insp_text = insp["observation"]
        therm_text = therm["observation"]

        if len(therm_text) > len(insp_text):
            combined_obs = f"{therm_text} [Inspection also notes: {insp_text}]"
        else:
            combined_obs = f"{insp_text} [Thermal confirms: {therm_text}]"

        # Use the more specific area name
        area = insp["area"] if len(insp["area"]) >= len(therm["area"]) else therm["area"]

        # Take the higher severity
        severity = self._higher_severity(
            insp.get("severity_hint", ""),
            therm.get("severity_hint", ""),
        )

        # Combine recommendations
        rec_parts = []
        if insp.get("recommendation_hint"):
            rec_parts.append(insp["recommendation_hint"])
        if therm.get("recommendation_hint"):
            if therm["recommendation_hint"] not in rec_parts:
                rec_parts.append(therm["recommendation_hint"])
        recommendation = "; ".join(rec_parts)

        # Collect sources
        sources = []
        if insp.get("source"):
            sources.append(insp["source"])
        if therm.get("source"):
            sources.append(therm["source"])

        pages = []
        if insp.get("page"):
            pages.append(f"{insp['source']} p.{insp['page']}")
        if therm.get("page"):
            pages.append(f"{therm['source']} p.{therm['page']}")

        # Corroborated findings get a confidence boost
        confidence = min(0.95, similarity * 0.5 + 0.5)

        image_refs = []
        if therm.get("image_reference"):
            image_refs.append(therm["image_reference"])

        return MergedObservation(
            area=area,
            observation=combined_obs,
            severity=severity,
            recommendation=recommendation,
            sources=sources,
            pages=pages,
            confidence_score=round(confidence, 3),
            merge_method="semantic",
            temperature_data=therm.get("temperature_data", ""),
            thermal_pattern=therm.get("thermal_pattern", ""),
            image_references=image_refs,
            is_corroborated=True,
        )

    def _single_to_merged(self, item: dict) -> MergedObservation:
        """Convert a single unmatched observation to MergedObservation."""
        source_type = item.get("_source_type", "unknown")

        image_refs = []
        if item.get("image_reference"):
            image_refs.append(item["image_reference"])

        return MergedObservation(
            area=item.get("area", ""),
            observation=item.get("observation", ""),
            severity=item.get("severity_hint", ""),
            recommendation=item.get("recommendation_hint", ""),
            sources=[item.get("source", "")] if item.get("source") else [],
            pages=[f"{item['source']} p.{item['page']}"] if item.get("page") else [],
            confidence_score=0.7 if source_type == "inspection" else 0.65,
            merge_method=f"single_{source_type}",
            temperature_data=item.get("temperature_data", ""),
            thermal_pattern=item.get("thermal_pattern", ""),
            image_references=image_refs,
            is_corroborated=False,
        )

    def _higher_severity(self, sev1: str, sev2: str) -> str:
        """Return the more severe of two severity levels."""
        order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "informational": 4}
        s1 = sev1.lower().strip() if sev1 else ""
        s2 = sev2.lower().strip() if sev2 else ""

        p1 = order.get(s1, 99)
        p2 = order.get(s2, 99)

        if p1 <= p2:
            return s1 if s1 else s2
        return s2 if s2 else s1


def merge_observations(inspection_obs: list, thermal_obs: list) -> MergeResult:
    """
    Convenience function for merging observation lists.

    Args:
        inspection_obs: Observations from inspection report.
        thermal_obs: Observations from thermal report.

    Returns:
        MergeResult with all merged observations.
    """
    merger = ObservationMerger()
    return merger.merge(inspection_obs, thermal_obs)

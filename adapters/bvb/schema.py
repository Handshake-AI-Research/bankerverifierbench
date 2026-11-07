"""Typed views over the BVB v1.0 release manifests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Criterion:
    """A scoring-eligible canonical evaluation criterion."""

    evaluation_index: int
    source_index: int
    text: str
    text_sha256: str
    category: str
    weight: int | float


@dataclass(frozen=True)
class SelectedRollout:
    """The one human-reviewed rollout selected for this benchmark task."""

    model: str
    trial_name: str
    candidate_path: str
    candidate_tree_sha256: str


@dataclass(frozen=True)
class BVBTask:
    """One task materialized from the immutable release manifest."""

    task_id: str
    task_record: dict[str, Any]
    manifest: dict[str, Any]
    criteria: tuple[Criterion, ...]
    selected_rollout: SelectedRollout | None
    labels_path: str | None

    @property
    def is_adjudicated(self) -> bool:
        return self.selected_rollout is not None and bool(self.criteria) and self.labels_path is not None

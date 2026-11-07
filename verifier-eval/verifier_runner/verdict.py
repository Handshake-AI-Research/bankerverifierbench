"""Canonical verifier-verdict schema.

One ``Verdict`` per (verifier, task, rollout) triple. Each verifier's native
output is translated into this shape by its runner before scoring. The schema
is deliberately polarity-natural: ``CriterionVerdict.satisfied = True`` means
the verifier said the agent met the criterion. Any inversion to UNMET-as-
positive happens in :mod:`verifier_runner.scoring`, not here.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TokenUsage(BaseModel):
    """Token counts. Always populated; zero where the verifier doesn't report.

    Cached reads and writes are subsets of ``prompt_tokens``; uncached input
    is ``prompt_tokens - cached_prompt_tokens - cache_write_prompt_tokens``.
    Reasoning tokens are a subset of completion tokens, not an extra output.
    """

    prompt_tokens: int = 0
    cached_prompt_tokens: int = 0
    cache_write_prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_completion_tokens: int = 0


class CriterionVerdict(BaseModel):
    """One verifier judgment for one rubric criterion.

    Index-aligned to ``rubric.json``: ``index`` is the criterion's position
    in the rubric, and the same ``index`` is used to look up the matching
    ground-truth entry during scoring.
    """

    index: int
    evaluation_criterion_index: int | None = Field(
        default=None,
        description="Canonical BVB evaluation-rubric index.",
    )
    source_criterion_index: int | None = Field(
        default=None,
        description="Source-rubric index retained for post-hoc provenance.",
    )
    criterion_text_sha256: str | None = Field(
        default=None,
        description="SHA-256 of the canonical criterion wording.",
    )
    satisfied: bool = Field(
        description="True iff the verifier judged the criterion met. "
        "Polarity-natural; scoring inverts to UNMET-as-positive."
    )
    confidence: float | None = Field(
        default=None,
        description="Continuous score in [0,1] when the verifier produces one "
        "(e.g. archipelago). None for boolean-only verifiers.",
    )
    reason: str = ""
    raw_verdict: str | None = Field(
        default=None,
        description="The verifier's native verdict string (e.g. 'MET', "
        "'UNMET', 'CANNOT_ASSESS'). Preserved so threshold/abstention "
        "policy choices stay reversible.",
    )


VerifierName = Literal[
    "gandalf",
    "rewardkit",
    "archipelago",
    "archipelago-tool-state",
    "aaaj",
    "autorubric",
]


class Verdict(BaseModel):
    """Canonical per-task verifier output. Persisted as JSON."""

    schema_version: Literal["1.0"] = "1.0"
    verifier: VerifierName
    model: str
    task_id: str = Field(
        description="BVB task ID matching the generated task directory."
    )
    gt_task_id: str | None = Field(
        default=None,
        description="BVB release task ID used to load canonical human labels.",
    )
    rollout_id: str | None = None
    cohort: str | None = Field(
        default=None,
        description="BVB cohort; the release has one, 'bvb'.",
    )
    canonical_rubric_sha256: str | None = Field(
        default=None,
        description="Digest of the BVB canonical evaluation rubric.",
    )
    trajectory_provenance: str | None = Field(
        default=None,
        description="Whether the verdict used an authentic or synthetic trajectory.",
    )
    input_identity: dict | None = None
    criteria: list[CriterionVerdict]
    tokens: TokenUsage = Field(default_factory=TokenUsage)
    cost_usd: float = 0.0
    duration_sec: float = 0.0

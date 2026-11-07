"""Canonical BVB criterion identity stamping and validation."""

from __future__ import annotations

from .inputs import InputBundle
from .verdict import CriterionVerdict, Verdict


def stamp_criterion_identity(verdict: Verdict, bundle: InputBundle) -> Verdict:
    """Attach release-derived identities to normalized verifier output.

    Verifier integrations produce verdicts positionally. The generated BVB
    rubric is canonical and includes stable identity fields, so the orchestration
    boundary converts that transient positional output into immutable verdict
    identities before it reaches disk.
    """
    if len(verdict.criteria) != len(bundle.rubric):
        raise ValueError(
            f"{verdict.verifier}/{bundle.task_id} emitted {len(verdict.criteria)} verdicts "
            f"for {len(bundle.rubric)} canonical criteria"
        )
    criteria: list[CriterionVerdict] = []
    for position, criterion in enumerate(verdict.criteria):
        rubric = bundle.rubric[position]
        try:
            evaluation_index = int(rubric["evaluation_criterion_index"])
            source_index = int(rubric["source_criterion_index"])
            text_digest = str(rubric["criterion_text_sha256"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Generated rubric for {bundle.task_id} lacks canonical criterion identity"
            ) from error
        if criterion.index != position:
            raise ValueError(
                f"{verdict.verifier}/{bundle.task_id} emitted non-positional criterion index "
                f"{criterion.index} at position {position}"
            )
        criteria.append(
            criterion.model_copy(
                update={
                    "evaluation_criterion_index": evaluation_index,
                    "source_criterion_index": source_index,
                    "criterion_text_sha256": text_digest,
                }
            )
        )
    return verdict.model_copy(
        update={
            "criteria": criteria,
            "input_identity": bundle.input_identity,
            "canonical_rubric_sha256": bundle.canonical_rubric_sha256,
            "trajectory_provenance": bundle.trajectory_provenance,
        }
    )

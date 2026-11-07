"""Fail-closed BVB scoring with canonical criterion identities."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapters.bvb.contract import input_identity
from adapters.bvb.release import BVBRelease
from .verdict import Verdict


@dataclass(frozen=True)
class ConfusionCounts:
    """Classical TP/FP/FN/TN with UNMET as the positive class."""

    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    def __add__(self, other: "ConfusionCounts") -> "ConfusionCounts":
        return ConfusionCounts(
            tp=self.tp + other.tp,
            fp=self.fp + other.fp,
            fn=self.fn + other.fn,
            tn=self.tn + other.tn,
        )


@dataclass(frozen=True)
class Metrics:
    """Metrics computed from pooled criterion-level confusion counts."""

    counts: ConfusionCounts
    precision: float
    recall: float
    f1: float
    accuracy: float
    mcc: float
    unmet_rate: float


@dataclass(frozen=True)
class TaskScore:
    """One task's fully validated criterion-level score."""

    task_id: str
    counts: ConfusionCounts
    n_criteria: int
    n_predictions: int
    human_overall_score: float = 0.0
    verifier_overall_score: float = 0.0
    overall_score_difference: float = 0.0


@dataclass(frozen=True)
class VerifierScore:
    """One verifier's strict cohort score."""

    verifier: str
    model: str
    metrics: Metrics
    per_task: list[TaskScore] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)


def metrics_from_counts(counts: ConfusionCounts, n_unmet_pred: int) -> Metrics:
    """Compute UNMET-positive metrics, using zero for undefined denominators."""
    tp, fp, fn, tn = counts.tp, counts.fp, counts.fn, counts.tn
    total = counts.total
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / total if total else 0.0
    mcc_denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = (tp * tn - fp * fn) / math.sqrt(mcc_denominator) if mcc_denominator else 0.0
    return Metrics(
        counts=counts,
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=accuracy,
        mcc=mcc,
        unmet_rate=n_unmet_pred / total if total else 0.0,
    )


def load_ground_truth(release: BVBRelease, task_id: str) -> list[dict]:
    """Return scoring-eligible human labels in canonical evaluation order."""
    try:
        task = release.tasks[task_id]
    except KeyError as error:
        raise KeyError(f"Unknown BVB task {task_id}") from error
    if not task.is_adjudicated:
        raise ValueError(f"{task_id} cannot be scored: canonical labels or selected rollout missing")

    labels_by_source: dict[int, dict] = {}
    for label in release.canonical_labels(task).get("labels", []):
        if not label.get("scoring_eligible"):
            continue
        source_index = int(label["source_criterion_index"])
        if source_index in labels_by_source:
            raise ValueError(f"{task_id} has duplicate human labels at source index {source_index}")
        labels_by_source[source_index] = label

    labels: list[dict] = []
    for criterion in task.criteria:
        label = labels_by_source.get(criterion.source_index)
        if label is None:
            raise ValueError(
                f"{task_id} canonical criterion {criterion.evaluation_index} has no human label"
            )
        if label.get("canonical_criterion_text_sha256") != criterion.text_sha256:
            raise ValueError(
                f"{task_id} label text digest disagrees with canonical rubric at "
                f"source index {criterion.source_index}"
            )
        if label.get('canonical_criterion_text') != criterion.text:
            raise ValueError(f'{task_id} actual human criterion text differs from canonical rubric')
        decision = label.get("final_adjudicated_decision_normalized")
        if decision not in {"MET", "UNMET"}:
            raise ValueError(
                f"{task_id} criterion {criterion.evaluation_index} has non-scoring decision {decision!r}"
            )
        labels.append(
            {
                "evaluation_criterion_index": criterion.evaluation_index,
                "source_criterion_index": criterion.source_index,
                "criterion_text_sha256": criterion.text_sha256,
                "decision": decision,
            }
        )
    if len(labels) != len(task.criteria):
        raise ValueError(f"{task_id} has incomplete canonical human labels")
    return labels


def _validate_verdict_identity(release: BVBRelease, task_id: str, verdict: Verdict) -> None:
    """Reject missing, duplicate, shuffled, or cross-rubric predictions."""
    task = release.tasks[task_id]
    expected_digest = task.manifest["evaluation_rubric"]["sha256"]
    if verdict.input_identity != input_identity(release, task):
        raise ValueError(f"{task_id} verdict input identity differs from pinned human-reviewed rollout")
    if verdict.canonical_rubric_sha256 != expected_digest:
        raise ValueError(
            f"{task_id} rubric digest mismatch: expected {expected_digest}, "
            f"got {verdict.canonical_rubric_sha256}"
        )
    expected = {criterion.evaluation_index: criterion for criterion in task.criteria}
    actual = {}
    for criterion in verdict.criteria:
        index = criterion.evaluation_criterion_index
        if index is None or criterion.source_criterion_index is None or not criterion.criterion_text_sha256:
            raise ValueError(f"{task_id} prediction lacks canonical criterion identity")
        if index in actual:
            raise ValueError(f"{task_id} has duplicate prediction for criterion {index}")
        actual[index] = criterion
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise ValueError(f"{task_id} prediction coverage mismatch; missing={missing}, extra={extra}")
    for index, expected_criterion in expected.items():
        predicted = actual[index]
        if (
            predicted.index != index
            or predicted.source_criterion_index != expected_criterion.source_index
            or predicted.criterion_text_sha256 != expected_criterion.text_sha256
        ):
            raise ValueError(f"{task_id} prediction identity mismatch at evaluation index {index}")


def confusion_for_task(ground_truth: list[dict], verdict: Verdict) -> tuple[ConfusionCounts, int]:
    """Score a fully validated task without positional truncation."""
    if len(ground_truth) != len(verdict.criteria):
        raise ValueError(
            f"{verdict.task_id} has {len(verdict.criteria)} predictions for "
            f"{len(ground_truth)} canonical labels"
        )
    by_index = {criterion.evaluation_criterion_index: criterion for criterion in verdict.criteria}
    tp = fp = fn = tn = n_unmet_pred = 0
    for label in ground_truth:
        prediction = by_index.get(label["evaluation_criterion_index"])
        if prediction is None:
            raise ValueError(f"{verdict.task_id} lacks a canonical prediction")
        gt_unmet = label["decision"] == "UNMET"
        pred_unmet = not prediction.satisfied
        n_unmet_pred += int(pred_unmet)
        if gt_unmet and pred_unmet:
            tp += 1
        elif not gt_unmet and pred_unmet:
            fp += 1
        elif gt_unmet:
            fn += 1
        else:
            tn += 1
    return ConfusionCounts(tp=tp, fp=fp, fn=fn, tn=tn), n_unmet_pred


def score_verifier(
    verdicts: list[Verdict],
    release_dir: Path,
    *,
    cohort: str,
    expected_task_ids: list[str] | None = None,
) -> VerifierScore:
    """Score exactly the complete BVB adjudicated cohort."""
    if not verdicts:
        raise ValueError("score_verifier requires at least one verdict")
    release = BVBRelease.open(release_dir)
    if cohort != release.cohort:
        raise ValueError(f"Unknown BVB scoring cohort {cohort!r}")

    verifiers = {verdict.verifier for verdict in verdicts}
    models = {verdict.model for verdict in verdicts}
    if len(verifiers) != 1 or len(models) != 1:
        raise ValueError("A score must contain exactly one verifier and one model")
    by_task = {verdict.task_id: verdict for verdict in verdicts}
    if len(by_task) != len(verdicts):
        raise ValueError("Multiple verdicts exist for the same task")

    expected = expected_task_ids or release.task_ids(release.cohort)
    adjudicated = release.task_ids(release.cohort)
    if set(expected) != set(adjudicated):
        raise ValueError("Publication scoring requires exactly the release adjudicated cohort")
    missing = sorted(set(expected) - set(by_task))
    extra = sorted(set(by_task) - set(expected))
    if missing or extra:
        raise ValueError(f"Verdict task coverage mismatch; missing={missing}, extra={extra}")

    total_counts = ConfusionCounts()
    total_unmet_predictions = 0
    per_task: list[TaskScore] = []
    for task_id in sorted(expected):
        verdict = by_task[task_id]
        if verdict.cohort != cohort:
            raise ValueError(f"{task_id} belongs to verdict cohort {verdict.cohort!r}, not {cohort!r}")
        _validate_verdict_identity(release, task_id, verdict)
        ground_truth = load_ground_truth(release, task_id)
        counts, unmet_predictions = confusion_for_task(ground_truth, verdict)
        weights = [float(c.weight) for c in release.tasks[task_id].criteria]
        denominator = sum(weights)
        predictions = {c.evaluation_criterion_index: c.satisfied for c in verdict.criteria}
        human_overall = sum(w for w, label in zip(weights, ground_truth)
                            if label['decision'] == 'MET') / denominator
        verifier_overall = sum(w for i, w in enumerate(weights) if predictions[i]) / denominator
        total_counts = total_counts + counts
        total_unmet_predictions += unmet_predictions
        per_task.append(
            TaskScore(
                task_id=task_id,
                counts=counts,
                n_criteria=len(ground_truth),
                n_predictions=len(verdict.criteria),
                human_overall_score=human_overall,
                verifier_overall_score=verifier_overall,
                overall_score_difference=verifier_overall - human_overall,
            )
        )

    expected_count = sum(len(release.tasks[t].criteria) for t in adjudicated)
    if total_counts.total != expected_count:
        raise ValueError(f"{cohort} must score {expected_count} criteria, scored {total_counts.total}")
    return VerifierScore(
        verifier=verdicts[0].verifier,
        model=verdicts[0].model,
        metrics=metrics_from_counts(total_counts, total_unmet_predictions),
        per_task=per_task,
        skipped=[],
    )


def baseline_metrics(ground_truth_totals: ConfusionCounts) -> dict[str, Metrics]:
    """Compute trivial baselines from a ground-truth class distribution."""
    n_unmet = ground_truth_totals.tp
    n_met = ground_truth_totals.tn
    n_total = n_unmet + n_met
    return {
        "always-unmet": metrics_from_counts(
            ConfusionCounts(tp=n_unmet, fp=n_met), n_unmet_pred=n_total
        ),
        "always-met": metrics_from_counts(
            ConfusionCounts(fn=n_unmet, tn=n_met), n_unmet_pred=0
        ),
    }

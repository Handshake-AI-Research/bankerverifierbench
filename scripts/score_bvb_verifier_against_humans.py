#!/usr/bin/env python3
"""Score repeated BVB verifier judgments against adjudicated human labels."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "verifier-eval")]

from adapters.bvb.config import DEFAULT_RELEASE_DIR
from adapters.bvb.contract import file_hash, input_identity, tree_hash
from adapters.bvb.release import BVBRelease, sha256_file
from scripts.run_bvb_gandalf_daytona import validate_job
from verifier_runner.scoring import (
    ConfusionCounts,
    load_ground_truth,
    metrics_from_counts,
)


def _confusion(
    human_met: list[bool],
    predicted_met: list[bool],
) -> tuple[ConfusionCounts, int]:
    if len(human_met) != len(predicted_met):
        raise ValueError("Human and verifier decision counts differ")
    tp = fp = fn = tn = predicted_unmet = 0
    for human, predicted in zip(human_met, predicted_met, strict=True):
        human_unmet = not human
        verifier_unmet = not predicted
        predicted_unmet += int(verifier_unmet)
        if human_unmet and verifier_unmet:
            tp += 1
        elif human and verifier_unmet:
            fp += 1
        elif human_unmet:
            fn += 1
        else:
            tn += 1
    return ConfusionCounts(tp=tp, fp=fp, fn=fn, tn=tn), predicted_unmet


def _metric_dict(
    counts: ConfusionCounts,
    predicted_unmet: int,
) -> dict[str, Any]:
    metrics = metrics_from_counts(counts, predicted_unmet)
    human_met = counts.tn + counts.fp
    human_unmet = counts.tp + counts.fn
    tnr = counts.tn / human_met if human_met else 0.0
    balanced_accuracy = (metrics.recall + tnr) / 2
    precision_met = (
        counts.tn / (counts.tn + counts.fn)
        if counts.tn + counts.fn
        else 0.0
    )
    recall_met = (
        counts.tn / (counts.tn + counts.fp)
        if counts.tn + counts.fp
        else 0.0
    )
    f1_met = (
        2 * precision_met * recall_met / (precision_met + recall_met)
        if precision_met + recall_met
        else 0.0
    )
    observed = metrics.accuracy
    total = counts.total
    predicted_met = counts.fn + counts.tn
    expected = (
        ((human_unmet * predicted_unmet) + (human_met * predicted_met))
        / (total * total)
        if total
        else 0.0
    )
    kappa = (observed - expected) / (1 - expected) if expected < 1 else 0.0
    return {
        "counts": asdict(counts),
        "total": total,
        "precision_unmet": metrics.precision,
        "recall_unmet": metrics.recall,
        "f1_unmet": metrics.f1,
        "precision_met": precision_met,
        "recall_met": recall_met,
        "f1_met": f1_met,
        "macro_f1": (metrics.f1 + f1_met) / 2,
        "accuracy": metrics.accuracy,
        "balanced_accuracy": balanced_accuracy,
        "mcc": metrics.mcc,
        "cohen_kappa": kappa,
        "predicted_unmet_rate": metrics.unmet_rate,
        "human_unmet_rate": human_unmet / total if total else 0.0,
    }


def _weighted_accuracy(
    human_met: list[bool],
    predicted_met: list[bool],
    weights: list[float],
) -> float:
    if not (len(human_met) == len(predicted_met) == len(weights)):
        raise ValueError("Weighted decision vectors differ in length")
    total = sum(weights)
    return (
        sum(
            weight
            for human, predicted, weight in zip(
                human_met, predicted_met, weights, strict=True
            )
            if human == predicted
        )
        / total
        if total
        else 0.0
    )


def _majority(decisions: list[bool]) -> bool:
    if len(decisions) % 2 == 0:
        raise ValueError("Majority voting requires an odd number of decisions")
    return sum(decisions) > len(decisions) / 2


def _verified_human_sources(
    release: BVBRelease,
    task_ids: list[str] | None = None,
) -> list[dict[str, str]]:
    """Verify the final annotation files consumed by scoring, with no private sources."""
    manifest = dict(line.split("  ", 1)[::-1] for line in
                    (release.root / "manifests/files.sha256").read_text().splitlines())
    sources = []
    for task_id in task_ids or release.task_ids(release.cohort):
        relative = release.tasks[task_id].labels_path
        actual = sha256_file(release.root / relative)
        if actual != manifest.get(relative):
            raise ValueError(f"{task_id}: canonical annotation file differs from the pinned release")
        sources.append({"task_id": task_id, "path": relative, "sha256": actual, "status": "verified"})
    return sources


def _load_human_labels(
    release: BVBRelease,
    task_id: str,
) -> list[dict[str, Any]]:
    task = release.tasks[task_id]
    scoring_labels = load_ground_truth(release, task_id)
    canonical = release.canonical_labels(task)
    by_source = {
        int(label["source_criterion_index"]): label
        for label in canonical["labels"]
        if label.get("scoring_eligible")
    }
    labels = []
    for scoring, criterion in zip(
        scoring_labels,
        task.criteria,
        strict=True,
    ):
        raw = by_source[criterion.source_index]
        labels.append(
            {
                **scoring,
                "criterion": criterion.text,
                "category": criterion.category,
                "weight": float(criterion.weight),
                "human_met": scoring["decision"] == "MET",
                "criterion_status": raw.get("criterion_status"),
                "human_reviewer_disagreement": str(
                    raw.get("reviewer_disagreement_raw", "")
                ).strip(),
                "human_provenance": raw.get("provenance"),
                "reviewer_a": raw.get("reviewer_a"),
                "reviewer_b": raw.get("reviewer_b"),
            }
        )
    return labels


def _load_trials(
    job_dir: Path,
    release: BVBRelease,
    expected_task_ids: list[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    expected = set(expected_task_ids or release.task_ids(release.cohort))
    for trial_dir in sorted(path for path in job_dir.iterdir() if path.is_dir()):
        result_path = trial_dir / "result.json"
        info_path = trial_dir / "verifier" / "info.json"
        if not result_path.is_file() or not info_path.is_file():
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("exception_info") is not None:
            raise ValueError(f"{trial_dir.name}: trial has exception_info")
        task_id = str(result["task_name"])
        if task_id not in release.tasks:
            raise ValueError(f"Unknown task in verifier job: {task_id}")
        if task_id not in expected:
            continue
        info = json.loads(info_path.read_text(encoding="utf-8"))
        identity = input_identity(release, release.tasks[task_id])
        audit = json.loads((trial_dir / 'verifier/input-audit.json').read_text())
        if audit.get('status') != 'validated' or audit.get('identity') != identity:
            raise ValueError(f'{trial_dir.name}: selected rollout input audit mismatch')
        if (tree_hash(trial_dir / 'agent/workspace') != identity['workspace_sha256']
                or file_hash(trial_dir / 'agent/trajectory.json') != identity['trajectory_sha256']):
            raise ValueError(f'{trial_dir.name}: selected rollout workspace or trajectory mismatch')
        labels = _load_human_labels(release, task_id)
        criteria = info.get("criterion_results", [])
        if len(criteria) != len(labels):
            raise ValueError(
                f"{trial_dir.name}: {len(criteria)} predictions for "
                f"{len(labels)} canonical labels"
            )
        for index, (prediction, label) in enumerate(
            zip(criteria, labels, strict=True)
        ):
            if str(prediction.get("criterion")) != label["criterion"]:
                raise ValueError(
                    f"{trial_dir.name}: criterion text mismatch at index {index}"
                )
            if float(prediction.get("weight", 0)) != label["weight"]:
                raise ValueError(
                    f"{trial_dir.name}: criterion weight mismatch at index {index}"
                )
        by_task[task_id].append(
            {
                "trial": trial_dir.name,
                "reward": float(
                    result["verifier_result"]["rewards"]["reward"]
                ),
                "info_path": str(info_path),
                "predicted_met": [
                    bool(prediction["met"]) for prediction in criteria
                ],
            }
        )
    if set(by_task) != expected:
        raise ValueError(
            "Verifier job task coverage differs from the release cohort; "
            f"missing={sorted(expected - set(by_task))}, "
            f"extra={sorted(set(by_task) - expected)}"
        )
    attempts = {len(trials) for trials in by_task.values()}
    if len(attempts) != 1:
        raise ValueError(f"Tasks have inconsistent attempt counts: {attempts}")
    for trials in by_task.values():
        trials.sort(key=lambda trial: trial["trial"])
    return dict(sorted(by_task.items()))


def score(
    release_dir: Path,
    job_dir: Path,
    task_ids: list[str] | None = None,
) -> dict[str, Any]:
    release = BVBRelease.open(release_dir)
    expected_task_ids = task_ids or release.task_ids(release.cohort)
    by_task = _load_trials(job_dir, release, expected_task_ids)
    attempts = len(next(iter(by_task.values())))
    job_result = json.loads((job_dir / "result.json").read_text(encoding="utf-8"))
    if int(job_result.get("n_total_trials", -1)) == len(expected_task_ids) * attempts:
        validate_job(
            job_dir,
            task_ids=expected_task_ids,
            attempts=attempts,
        )
    sources = _verified_human_sources(release, expected_task_ids)

    pooled_counts = ConfusionCounts()
    pooled_predicted_unmet = 0
    pooled_human: list[bool] = []
    pooled_predictions: list[bool] = []
    pooled_weights: list[float] = []
    trial_scores = []
    per_task = {}
    criterion_rows = []
    majority_human: list[bool] = []
    majority_predictions: list[bool] = []
    majority_weights: list[float] = []
    majority_counts = ConfusionCounts()
    majority_predicted_unmet = 0
    reviewer_disagreement_groups: dict[str, dict[str, list[Any]]] = defaultdict(
        lambda: {"human": [], "predicted": [], "weights": []}
    )
    status_groups: dict[str, dict[str, list[Any]]] = defaultdict(
        lambda: {"human": [], "predicted": [], "weights": []}
    )

    for task_id, trials in by_task.items():
        labels = _load_human_labels(release, task_id)
        human = [bool(label["human_met"]) for label in labels]
        weights = [float(label["weight"]) for label in labels]
        task_counts = ConfusionCounts()
        task_predicted_unmet = 0
        for trial in trials:
            counts, predicted_unmet = _confusion(
                human,
                trial["predicted_met"],
            )
            task_counts += counts
            task_predicted_unmet += predicted_unmet
            pooled_counts += counts
            pooled_predicted_unmet += predicted_unmet
            pooled_human.extend(human)
            pooled_predictions.extend(trial["predicted_met"])
            pooled_weights.extend(weights)
            metrics = _metric_dict(counts, predicted_unmet)
            trial_scores.append(
                {
                    "task_id": task_id,
                    "trial": trial["trial"],
                    "reward": trial["reward"],
                    "info_path": trial["info_path"],
                    **metrics,
                    "weighted_accuracy": _weighted_accuracy(
                        human,
                        trial["predicted_met"],
                        weights,
                    ),
                }
            )

        majority = [
            _majority([trial["predicted_met"][index] for trial in trials])
            for index in range(len(labels))
        ]
        counts, predicted_unmet = _confusion(human, majority)
        majority_counts += counts
        majority_predicted_unmet += predicted_unmet
        majority_human.extend(human)
        majority_predictions.extend(majority)
        majority_weights.extend(weights)
        per_task[task_id] = {
            "attempts": attempts,
            "criteria": len(labels),
            "pooled_attempts": {
                **_metric_dict(task_counts, task_predicted_unmet),
                "weighted_accuracy": _weighted_accuracy(
                    human * attempts,
                    [
                        decision
                        for trial in trials
                        for decision in trial["predicted_met"]
                    ],
                    weights * attempts,
                ),
            },
            "majority_vote": {
                **_metric_dict(counts, predicted_unmet),
                "weighted_accuracy": _weighted_accuracy(
                    human,
                    majority,
                    weights,
                ),
            },
            "trial_metrics": [
                row for row in trial_scores if row["task_id"] == task_id
            ],
        }

        for index, label in enumerate(labels):
            predictions = [
                bool(trial["predicted_met"][index]) for trial in trials
            ]
            majority_prediction = majority[index]
            unanimous = len(set(predictions)) == 1
            majority_correct = majority_prediction == label["human_met"]
            if unanimous and majority_correct:
                stability = "unanimous_correct"
            elif unanimous:
                stability = "unanimous_wrong"
            elif majority_correct:
                stability = "flip_majority_correct"
            else:
                stability = "flip_majority_wrong"
            if label["human_met"] and not majority_prediction:
                error_type = "false_positive_unmet"
            elif not label["human_met"] and majority_prediction:
                error_type = "false_negative_unmet"
            else:
                error_type = "correct"
            criterion_rows.append(
                {
                    "task_id": task_id,
                    "evaluation_criterion_index": label[
                        "evaluation_criterion_index"
                    ],
                    "source_criterion_index": label[
                        "source_criterion_index"
                    ],
                    "criterion": label["criterion"],
                    "criterion_text_sha256": label[
                        "criterion_text_sha256"
                    ],
                    "category": label["category"],
                    "weight": label["weight"],
                    "criterion_status": label["criterion_status"],
                    "human_decision": "MET" if label["human_met"] else "UNMET",
                    "human_reviewer_disagreement": label[
                        "human_reviewer_disagreement"
                    ],
                    "human_provenance": label["human_provenance"],
                    "verifier_decisions": [
                        {
                            "trial": trial["trial"],
                            "decision": "MET" if predicted else "UNMET",
                        }
                        for trial, predicted in zip(
                            trials,
                            predictions,
                            strict=True,
                        )
                    ],
                    "verifier_met_count": sum(predictions),
                    "majority_decision": (
                        "MET" if majority_prediction else "UNMET"
                    ),
                    "stability": stability,
                    "error_type": error_type,
                }
            )
            disagreement_key = (
                "human_reviewers_disagreed"
                if label["human_reviewer_disagreement"].lower()
                in {"yes", "true", "1"}
                else "human_reviewers_agreed"
            )
            for grouping, key in (
                (reviewer_disagreement_groups, disagreement_key),
                (status_groups, str(label["criterion_status"])),
            ):
                grouping[key]["human"].append(label["human_met"])
                grouping[key]["predicted"].append(majority_prediction)
                grouping[key]["weights"].append(label["weight"])

    pooled_metrics = _metric_dict(
        pooled_counts,
        pooled_predicted_unmet,
    )
    pooled_metrics["weighted_accuracy"] = _weighted_accuracy(
        pooled_human,
        pooled_predictions,
        pooled_weights,
    )
    majority_metrics = _metric_dict(
        majority_counts,
        majority_predicted_unmet,
    )
    majority_metrics["weighted_accuracy"] = _weighted_accuracy(
        majority_human,
        majority_predictions,
        majority_weights,
    )

    def grouped_scores(
        groups: dict[str, dict[str, list[Any]]],
    ) -> dict[str, Any]:
        output = {}
        for key, values in sorted(groups.items()):
            counts, predicted_unmet = _confusion(
                values["human"],
                values["predicted"],
            )
            output[key] = {
                **_metric_dict(counts, predicted_unmet),
                "weighted_accuracy": _weighted_accuracy(
                    values["human"],
                    values["predicted"],
                    values["weights"],
                ),
            }
        return output

    human_unmet = sum(not decision for decision in majority_human)
    human_met = len(majority_human) - human_unmet
    baselines = {
        "always_unmet": _metric_dict(
            ConfusionCounts(tp=human_unmet, fp=human_met),
            len(majority_human),
        ),
        "always_met": _metric_dict(
            ConfusionCounts(fn=human_unmet, tn=human_met),
            0,
        ),
    }
    stability_counts = Counter(row["stability"] for row in criterion_rows)
    error_counts = Counter(row["error_type"] for row in criterion_rows)
    excluded_criteria = []
    for task_id in expected_task_ids:
        canonical = release.canonical_labels(release.tasks[task_id])
        for excluded in canonical["evaluation_rubric"].get(
            "excluded_criteria", []
        ):
            excluded_criteria.append(
                {
                    "task_id": task_id,
                    "source_criterion_index": excluded.get(
                        "source_criterion_index"
                    ),
                    "criterion_status": excluded.get("criterion_status"),
                    "final_adjudicated_decision_raw": excluded.get(
                        "final_adjudicated_decision_raw"
                    ),
                    "reason": excluded.get("reason"),
                }
            )
    return {
        "schema_version": "bvb-v1.0-human-score/v1",
        "release_dir": str(release_dir),
        "job_dir": str(job_dir),
        "cohort": release.cohort,
        "task_filter": expected_task_ids,
        "tasks": len(by_task),
        "attempts_per_task": attempts,
        "criteria": len(majority_human),
        "predictions": len(pooled_predictions),
        "human_label_source": {
            "policy": "Final canonical annotations verified against the pinned release checksum manifest.",
            "files": sources,
        },
        "excluded_criteria": {
            "count": len(excluded_criteria),
            "policy": (
                "DELETED_POSTHOC and MOOT rows are preserved for provenance "
                "and excluded from all metrics"
            ),
            "rows": excluded_criteria,
        },
        "human_class_distribution": {
            "met": human_met,
            "unmet": human_unmet,
            "unmet_rate": human_unmet / len(majority_human),
        },
        "pooled_attempts": pooled_metrics,
        "majority_vote": majority_metrics,
        "baselines": baselines,
        "trial_metric_distribution": {
            metric: {
                "mean": statistics.fmean(
                    float(row[metric]) for row in trial_scores
                ),
                "median": statistics.median(
                    float(row[metric]) for row in trial_scores
                ),
                "min": min(float(row[metric]) for row in trial_scores),
                "max": max(float(row[metric]) for row in trial_scores),
            }
            for metric in (
                "accuracy",
                "balanced_accuracy",
                "f1_unmet",
                "f1_met",
                "macro_f1",
                "mcc",
                "cohen_kappa",
                "weighted_accuracy",
            )
        },
        "criterion_stability_against_humans": dict(stability_counts),
        "majority_error_types": dict(error_counts),
        "human_reviewer_disagreement_scores": grouped_scores(
            reviewer_disagreement_groups
        ),
        "criterion_status_scores": grouped_scores(status_groups),
        "per_task": per_task,
        "trials": trial_scores,
        "criteria_detail": criterion_rows,
    }


def _write_csvs(output: Path, result: dict[str, Any]) -> None:
    trial_path = output.with_suffix(".trials.csv")
    trial_fields = [
        "task_id",
        "trial",
        "reward",
        "accuracy",
        "balanced_accuracy",
        "precision_unmet",
        "recall_unmet",
        "f1_unmet",
        "f1_met",
        "macro_f1",
        "mcc",
        "cohen_kappa",
        "weighted_accuracy",
        "predicted_unmet_rate",
        "human_unmet_rate",
        "TP",
        "FP",
        "FN",
        "TN",
        "info_path",
    ]
    with trial_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=trial_fields)
        writer.writeheader()
        for row in result["trials"]:
            writer.writerow(
                {
                    **{
                        key: row.get(key)
                        for key in trial_fields
                        if key not in {"TP", "FP", "FN", "TN"}
                    },
                    "TP": row["counts"]["tp"],
                    "FP": row["counts"]["fp"],
                    "FN": row["counts"]["fn"],
                    "TN": row["counts"]["tn"],
                }
            )

    criteria_path = output.with_suffix(".criteria.csv")
    criteria_fields = [
        "task_id",
        "evaluation_criterion_index",
        "source_criterion_index",
        "category",
        "weight",
        "criterion_status",
        "human_decision",
        "human_reviewer_disagreement",
        "verifier_met_count",
        "majority_decision",
        "stability",
        "error_type",
        "criterion",
    ]
    with criteria_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=criteria_fields)
        writer.writeheader()
        writer.writerows(
            {
                key: row.get(key)
                for key in criteria_fields
            }
            for row in result["criteria_detail"]
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE_DIR)
    parser.add_argument("--job-dir", type=Path, required=True, help="Harbor job directory to score.")
    parser.add_argument("--output", type=Path, required=True, help="Score JSON; CSVs are written alongside.")
    parser.add_argument(
        "--task-ids",
        help="Optional comma-separated adjudicated task subset.",
    )
    args = parser.parse_args()
    task_ids = (
        [value.strip() for value in args.task_ids.split(",") if value.strip()]
        if args.task_ids
        else None
    )
    result = score(
        args.release_dir.resolve(),
        args.job_dir.resolve(),
        task_ids,
    )
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csvs(args.output, result)
    print(
        json.dumps(
            {
                "pooled_attempts": result["pooled_attempts"],
                "majority_vote": result["majority_vote"],
                "human_class_distribution": result[
                    "human_class_distribution"
                ],
                "criterion_stability_against_humans": result[
                    "criterion_stability_against_humans"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

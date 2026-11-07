"""Score a verifier's criterion decisions against the BVB human labels.

Predictions are JSON Lines with one object per criterion and repeat:

    {"task_id": "btb-4fedc1aa", "evaluation_criterion_index": 0,
     "criterion_text_sha256": "eb3013fa...", "repeat": 1, "decision": "MET"}

`repeat` defaults to 1 and other keys are ignored. Each (task, repeat) present must decide every
scoring-eligible criterion of that task. Metrics use the paper's definitions (UNMET is the
positive class, decisions are pooled over tasks and repeats). They are comparable to
results/paper/conditions.csv only when all 24 tasks are covered for every repeat; majority-vote F1
and its task-bootstrap interval are reported for exactly three repeats, as in the paper.

`bvb score --verdicts <run output>` reads the Verdict files written by `bvb run` instead.
"""
from __future__ import annotations

import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from pydantic import ValidationError

from scripts.bvb_public_results import CONDITION_BOOTSTRAP, bootstrap_interval, scoring_metrics, summary_metrics, total

from .release import BVBRelease

ROOT = Path(__file__).resolve().parents[2]


class ScoreError(ValueError):
    pass


def load_predictions(path: Path) -> dict[tuple[str, int, int], tuple[str, str]]:
    """Map (task_id, evaluation_criterion_index, repeat) to (criterion_text_sha256, decision)."""
    predictions = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        where = f"{path}:{number}"
        try:
            row = json.loads(line)
            key = (str(row["task_id"]), int(row["evaluation_criterion_index"]), int(row.get("repeat", 1)))
            digest, decision = str(row["criterion_text_sha256"]), str(row["decision"]).upper()
        except json.JSONDecodeError as error:
            raise ScoreError(f"{where}: invalid JSON ({error})") from error
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            raise ScoreError(f"{where}: expected task_id, evaluation_criterion_index, "
                             f"criterion_text_sha256 and decision ({error!r})") from error
        if decision not in {"MET", "UNMET"}:
            raise ScoreError(f"{where}: decision must be MET or UNMET, got {row['decision']!r}")
        if key[2] < 1:
            raise ScoreError(f"{where}: repeat must be 1 or greater")
        if key in predictions:
            raise ScoreError(f"{where}: duplicate prediction for {key}")
        predictions[key] = (digest, decision)
    if not predictions:
        raise ScoreError(f"{path} contains no predictions")
    return predictions


def _native_verdicts(release: BVBRelease, paths: list[Path]) -> list[tuple[Path, object]]:
    """Parse and fail-closed validate Verdict files produced by ``bvb run``."""
    if str(ROOT / "verifier-eval") not in sys.path:
        sys.path.insert(0, str(ROOT / "verifier-eval"))
    from verifier_runner.scoring import _validate_verdict_identity
    from verifier_runner.verdict import Verdict

    verdicts = []
    configurations = set()
    for path in paths:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ScoreError(f"{path}: Verdict must be a JSON object")
            if raw.get("excluded") is True or raw.get("scoring_eligible") is False:
                raise ScoreError(f"{path}: excluded verdict cannot be scored")
            criteria = raw.get("criteria", [])
            if not isinstance(criteria, list):
                raise ScoreError(f"{path}: Verdict criteria must be a list")
            for criterion in criteria:
                if not isinstance(criterion, dict):
                    raise ScoreError(f"{path}: Verdict criterion must be an object")
                if (criterion.get("excluded") is True
                        or criterion.get("excluded_from_scoring") is True
                        or criterion.get("scoring_eligible") is False):
                    raise ScoreError(f"{path}: excluded criterion cannot be scored")
            verdict = Verdict.model_validate(raw, strict=True)
        except (OSError, json.JSONDecodeError) as error:
            raise ScoreError(f"{path}: invalid JSON ({error})") from error
        except ValidationError as error:
            raise ScoreError(f"{path}: invalid Verdict schema ({error})") from error
        if verdict.task_id not in release.tasks:
            raise ScoreError(f"{path}: unknown task {verdict.task_id!r}")
        if verdict.cohort != release.cohort:
            raise ScoreError(f"{path}: verdict cohort {verdict.cohort!r} is not {release.cohort!r}")
        try:
            _validate_verdict_identity(release, verdict.task_id, verdict)
        except (KeyError, ValueError) as error:
            raise ScoreError(f"{path}: native verdict identity is invalid ({error})") from error
        configurations.add((verdict.verifier, verdict.model, verdict.cohort))
        verdicts.append((path, verdict))
    if len(configurations) != 1:
        raise ScoreError("Verdict files mix verifier, model, or cohort; point --verdicts at one run configuration")
    return verdicts


def load_verdicts(run_dir: Path, release: BVBRelease) -> dict[tuple[str, int, int], tuple[str, str]]:
    """Predictions from the Verdict files a `bvb run` output holds (any runner except M01).

    Runners write verdicts/<task>.json or <task>/verdict.json below a repeat-<N> directory.
    """
    predictions = {}
    paths = sorted({*run_dir.rglob("verdicts/*.json"), *run_dir.rglob("verdict.json")})
    if not paths:
        raise ScoreError(f"No verdict files under {run_dir}")
    for path, verdict in _native_verdicts(release, paths):
        repeat = next((p.name.removeprefix("repeat-") for p in path.parents if p.name.startswith("repeat-")), None)
        if repeat is None or not repeat.isdigit():
            raise ScoreError(f"{path} is not below a repeat-<N> directory")
        for criterion in verdict.criteria:
            key = (verdict.task_id, int(criterion.evaluation_criterion_index), int(repeat))
            if key in predictions:
                raise ScoreError(f"{path}: duplicate verdict for {key}; point --verdicts at one verifier and model")
            predictions[key] = (str(criterion.criterion_text_sha256), "MET" if criterion.satisfied else "UNMET")
    return predictions


def _human_labels(release: BVBRelease) -> dict[str, list[tuple[str, float, str]]]:
    """Per task, (criterion_text_sha256, weight, human decision) in evaluation order."""
    if str(ROOT / "verifier-eval") not in sys.path:
        sys.path.insert(0, str(ROOT / "verifier-eval"))
    from verifier_runner.scoring import load_ground_truth

    labels = {}
    for task in release.select_tasks():
        truth = load_ground_truth(release, task.task_id)
        labels[task.task_id] = [(c.text_sha256, float(c.weight), label["decision"])
                                for c, label in zip(task.criteria, truth)]
    return labels


def score(release: BVBRelease, predictions: dict[tuple[str, int, int], tuple[str, str]]) -> dict:
    labels = _human_labels(release)
    runs: dict[tuple[str, int], dict[int, str]] = defaultdict(dict)
    for (task_id, index, repeat), (digest, decision) in predictions.items():
        if task_id not in labels:
            raise ScoreError(f"Unknown task {task_id!r}")
        if not 0 <= index < len(labels[task_id]):
            raise ScoreError(f"{task_id} has no evaluation criterion {index}")
        if digest != labels[task_id][index][0]:
            raise ScoreError(f"{task_id} criterion {index}: criterion_text_sha256 does not match the release")
        runs[task_id, repeat][index] = decision

    counts, abs_errors = {}, []
    for (task_id, repeat), decisions in sorted(runs.items()):
        criteria = labels[task_id]
        if len(decisions) != len(criteria):
            raise ScoreError(f"{task_id} repeat {repeat} decides {len(decisions)} of {len(criteria)} criteria")
        cc = Counter({k: 0 for k in ("tp", "fp", "fn", "tn")})
        weight = human_met = predicted_met = 0.0
        for index, (_, w, human) in enumerate(criteria):
            predicted = decisions[index]
            cc[("t" if predicted == human else "f") + ("p" if predicted == "UNMET" else "n")] += 1
            weight += w
            human_met += w * (human == "MET")
            predicted_met += w * (predicted == "MET")
        counts[task_id, repeat] = cc
        abs_errors.append(abs(predicted_met - human_met) / weight)

    tasks = sorted(labels)
    repeats = sorted({repeat for _, repeat in runs})
    complete = set(runs) == {(t, r) for t in tasks for r in range(1, len(repeats) + 1)}
    pooled = summary_metrics(total(counts.values()))
    result = dict(
        tasks=len({t for t, _ in runs}), of_tasks=len(tasks), repeats=repeats, runs=len(runs), complete=complete,
        decisions=sum(sum(cc.values()) for cc in counts.values()),
        **{k: pooled[k] for k in ("counts", "f1", "precision", "recall", "accuracy")},
        overall_mae=statistics.mean(abs_errors), majority_f1=None, ci_low=None, ci_high=None,
    )
    if complete and repeats == [1, 2, 3]:
        votes = Counter({k: 0 for k in ("tp", "fp", "fn", "tn")})
        for task_id in tasks:
            for index, (_, _, human) in enumerate(labels[task_id]):
                vote = Counter(runs[task_id, r][index] for r in repeats).most_common(1)[0][0]
                votes[("t" if vote == human else "f") + ("p" if vote == "UNMET" else "n")] += 1
        result["majority_f1"] = scoring_metrics(votes)["f1_unmet"]
        by_task = {t: total(counts[t, r] for r in repeats) for t in tasks}
        rng, draws = random.Random(CONDITION_BOOTSTRAP["seed"]), []
        for _ in range(CONDITION_BOOTSTRAP["draws"]):
            draws.append(scoring_metrics(total(by_task[t] for t in rng.choices(tasks, k=len(tasks))))["f1_unmet"])
        result["ci_low"], result["ci_high"] = bootstrap_interval(draws, CONDITION_BOOTSTRAP["draws"])
    return result


def report(result: dict, references: dict[str, dict]) -> str:
    coverage = (f"{result['tasks']}/{result['of_tasks']} tasks, repeats {','.join(map(str, result['repeats']))}, "
                f"{result['decisions']} decisions")
    lines = [f"{'coverage':11} {coverage}"
             + ("" if result["complete"] else "  (partial: not comparable to the paper table)")]
    for key in ("f1", "precision", "recall", "accuracy", "overall_mae"):
        lines.append(f"{key:11} {result[key]:.3f}")
    if result["majority_f1"] is not None:
        lines.append(f"{'majority_f1':11} {result['majority_f1']:.3f}  "
                     f"(F1 95% task-bootstrap interval {result['ci_low']:.3f}-{result['ci_high']:.3f})")
    lines += ["", "Paper reference (results/paper/conditions.csv):"]
    for cid, row in references.items():
        lines.append(f"  {cid:4} {row['verifier']:12} {row['model']:30} F1 {float(row['f1']):.3f}  "
                     f"MAE {float(row['overall_mae']):.3f}")
    return "\n".join(lines)

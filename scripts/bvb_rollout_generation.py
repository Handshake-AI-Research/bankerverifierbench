#!/usr/bin/env python3
"""Build or check the generation statistics of the 24 released BVB rollouts.

`build` reads each task's selected rollout trajectory from a downloaded release
and writes per-rollout usage (steps, tool calls, tokens, recorded API cost) to
`results/rollout-generation/`, with per-model summaries and the ratio of each
complete verifier condition's cost per dataset pass to the generation cost.

`check` uses the standard library only and makes no network or model calls. It
confirms every file hash, then recomputes the summary from `rollouts.csv` and
`results/paper/conditions.csv`. With `--release` it also re-extracts
`rollouts.csv` from the release trajectories.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results/rollout-generation"
CONDITIONS = ROOT / "results/paper/conditions.csv"
SCHEMA = "bvb-rollout-generation/v1"
MODEL_NAMES = {"gemini25-pro": "Gemini 2.5 Pro", "glm5": "GLM-5"}
FIELDS = ("task", "model", "model_route", "harness", "trajectory_sha256", "steps", "tool_calls",
          "prompt_tokens", "cached_tokens", "completion_tokens", "cost_usd")
INTEGER_FIELDS = ("steps", "tool_calls", "prompt_tokens", "cached_tokens", "completion_tokens")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def extract(release_dir: Path) -> list[dict]:
    sys.path.insert(0, str(ROOT))
    from adapters.bvb.release import BVBRelease

    release = BVBRelease.open(release_dir)
    release.assert_pinned_revision()
    rows = []
    for task_id in sorted(release.tasks):
        task = release.tasks[task_id]
        trajectory = task.manifest["replay_artifacts"]["trajectory"]
        payload = (release.root / trajectory["path"]).read_bytes()
        require(sha(payload) == trajectory["sha256"], f"{task_id}: trajectory hash differs from the manifest")
        data = json.loads(payload)
        totals = data["final_metrics"]
        require(all(totals.get(k) is not None for k in ("total_prompt_tokens", "total_cached_tokens",
                                                       "total_completion_tokens", "total_cost_usd", "total_steps")),
                f"{task_id}: trajectory usage is incomplete")
        rows.append(dict(
            task=task_id,
            model=task.selected_rollout.model,
            model_route=data["agent"]["model_name"],
            harness=f"{data['agent']['name']} {data['agent']['version']}",
            trajectory_sha256=trajectory["sha256"],
            steps=totals["total_steps"],
            tool_calls=sum(len(step.get("tool_calls") or []) for step in data["steps"]),
            prompt_tokens=totals["total_prompt_tokens"],
            cached_tokens=totals["total_cached_tokens"],
            completion_tokens=totals["total_completion_tokens"],
            cost_usd=totals["total_cost_usd"],
        ))
    return rows


def rollouts_csv(rows: list[dict]) -> str:
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def read_rollouts(text: str) -> list[dict]:
    rows = list(csv.DictReader(io.StringIO(text)))
    for row in rows:
        for key in INTEGER_FIELDS:
            row[key] = int(row[key])
        row["cost_usd"] = float(row["cost_usd"])
    return rows


def _group(rows: list[dict]) -> dict:
    calls = [r["tool_calls"] for r in rows]
    costs = [r["cost_usd"] for r in rows]
    prompt = sum(r["prompt_tokens"] for r in rows)
    top = max(rows, key=lambda r: r["cost_usd"])
    return dict(
        rollouts=len(rows),
        tool_calls=dict(mean=statistics.mean(calls), median=statistics.median(calls), min=min(calls), max=max(calls)),
        steps_mean=statistics.mean(r["steps"] for r in rows),
        prompt_tokens=prompt,
        cached_tokens=sum(r["cached_tokens"] for r in rows),
        completion_tokens=sum(r["completion_tokens"] for r in rows),
        cache_share=sum(r["cached_tokens"] for r in rows) / prompt,
        cost_usd=dict(total=sum(costs), mean=statistics.mean(costs), median=statistics.median(costs),
                      max=top["cost_usd"], max_task=top["task"], max_share=top["cost_usd"] / sum(costs)),
    )


def summary(rows: list[dict], conditions_text: str) -> dict:
    tasks = len(rows)
    generation = sum(r["cost_usd"] for r in rows)
    groups = {MODEL_NAMES[m]: _group([r for r in rows if r["model"] == m]) for m in sorted({r["model"] for r in rows})}
    groups["All"] = _group(rows)
    verification = []
    for c in csv.DictReader(io.StringIO(conditions_text)):
        if c["status"] != "Complete" or c["scope"] != "Selected":
            continue
        per_pass = float(c["total_cost"]) * tasks / int(c["runs"])
        verification.append(dict(id=c["id"], verifier=c["verifier"], model=c["model"], partition=c["partition"],
                                 cost_per_pass_usd=per_pass, ratio_to_generation=per_pass / generation))
    return dict(
        cost_basis=("Recorded model API cost in each trajectory's final_metrics, as reported via OpenRouter "
                    "and reflecting prompt-caching discounts; not an invoice."),
        tool_calls=("Tool calls recorded in the trajectory steps."),
        verification_basis=("Complete, selected conditions in results/paper/conditions.csv: total recorded API "
                            "cost scaled to one pass over the 24 tasks."),
        groups=groups,
        verification=verification,
    )


def summary_json(rows: list[dict], conditions_text: str) -> str:
    return json.dumps(summary(rows, conditions_text), indent=2) + "\n"


def check(output: Path = DEFAULT_OUTPUT, release_dir: Path | None = None) -> dict:
    manifest = json.loads((output / "manifest.json").read_text())
    require(manifest["schema"] == SCHEMA, "Unknown bundle schema")
    for record in manifest["files"]:
        payload = (output / record["path"]).read_bytes()
        require(len(payload) == record["bytes"] and sha(payload) == record["sha256"], f"File changed: {record['path']}")
    present = {p.name for p in output.iterdir() if p.is_file() and p.name not in ("manifest.json", "README.md")}
    require(present == {r["path"] for r in manifest["files"]}, "Unmanifested or missing files")
    conditions_text = CONDITIONS.read_text()
    require(sha(conditions_text.encode()) == manifest["inputs"]["conditions_csv_sha256"],
            "results/paper/conditions.csv differs from the one these ratios were built from")
    rows_text = (output / "rollouts.csv").read_text()
    rows = read_rollouts(rows_text)
    require(len(rows) == manifest["rollouts"], "Rollout count")
    require((output / "summary.json").read_text() == summary_json(rows, conditions_text),
            "summary.json does not match its recomputation")
    if release_dir is not None:
        require(rollouts_csv(extract(release_dir)) == rows_text, "rollouts.csv differs from the release trajectories")
    return dict(status="passed", rollouts=len(rows), bound_to_release=release_dir is not None, paid_model_calls=0)


def build(release_dir: Path, output: Path = DEFAULT_OUTPUT) -> dict:
    sys.path.insert(0, str(ROOT))
    from adapters.bvb.config import DATASET_MANIFEST_SHA256, DATASET_REPOSITORY, DATASET_REVISION

    rows = extract(release_dir)
    conditions_text = CONDITIONS.read_text()
    output.mkdir(parents=True, exist_ok=True)
    files = {"rollouts.csv": rollouts_csv(rows), "summary.json": summary_json(rows, conditions_text)}
    for name, text in files.items():
        (output / name).write_text(text)
    manifest = dict(
        schema=SCHEMA,
        dataset_repository=DATASET_REPOSITORY,
        dataset_revision=DATASET_REVISION,
        dataset_manifest_sha256=DATASET_MANIFEST_SHA256,
        rollouts=len(rows),
        builder_sha256=sha(Path(__file__).read_bytes()),
        inputs=dict(conditions_csv_sha256=sha(conditions_text.encode())),
        files=[dict(path=name, bytes=len(text.encode()), sha256=sha(text.encode())) for name, text in files.items()],
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return check(output, release_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    check_parser = sub.add_parser("check")
    check_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    check_parser.add_argument("--release", type=Path, help="Also re-extract rollouts.csv from this public release.")
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--release", type=Path, required=True, help="Downloaded public release.")
    build_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = check(args.output, args.release) if args.command == "check" else build(args.release, args.output)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

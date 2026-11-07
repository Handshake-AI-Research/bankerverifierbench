# BVB verifier evaluation

`verifier_runner` stages generated BankerVerifierBench Harbor tasks, runs native
verifiers on the staged evidence, normalizes their output and computes strict
human-label metrics. Start with the [root README](../README.md) and the
[reproduction guide](../docs/REPRODUCING_BVB.md).

## Inputs

The release is the pinned BankerVerifierBench dataset revision (24 tasks,
2,464 scoring-eligible criteria) fetched by `make fetch`. Its checksum manifest
is pinned in code, so a local copy cannot silently change prompts, rubrics,
labels or rollout selection.

Judge criteria come exclusively from `manifests/evaluation-rubrics/`.
The original `task-source/*/tests/rubric.json` and archived verifier outputs
are historical evidence and may contain earlier criterion wording. Excluded
criteria are omitted. The prompt comes from the frozen
`task-source/<task-id>/instruction.md`, and the selected human-reviewed rollout
supplies the complete authentic `agent/workspace` and `agent/trajectory.json`.

Materialization, host judge entry points, runtime preflight and scoring reject
mismatches in actual text, weights, ordering and input identity. Each new result
records the release revision, rubric hash, prompt hash, selected model/trial,
workspace hash and trajectory hash. Runtime `input-audit.json` records the
original files validated by the trusted outer verifier. Gandalf's separate
`mcp-audit.json` records the inner sandbox UID, groups and available tools;
the outer audit requires UID 10003 with no agent-group membership. Host `judge-inputs.json`
records the exact common judge bundle. Gandalf and RewardKit runtime rubric
files are checked independently. The runtime preflight also probes the exact
configured MCP launch command with a filtered environment and requires all
15 tools before any judge call. Lazy archive settings are passed explicitly
for Docker and Daytona; the inner judge does not inherit them automatically.

Both runtimes expose the full replay under `/home/verifier/review-inputs`,
using real directories and file symlinks: `workspace/` and `trajectory.json`.
Gandalf's outer judge copies these files into each inner judge's
`/tmp/judge_workspace_*`; directory symlinks would be skipped by its copier.
The inner judge keeps the original sandbox role: it accesses those copies,
not the original agent workspace. The Dockerfile only adds an environment-owned
private cache outside the read-only MCP archive mounts.
The guidance identifies the relative paths and asks the judge to inspect them.
Gandalf automatically injects only the final-response excerpt, so the full
trajectory is read through filesystem tools rather than inserted into the
initial model prompt. Source-data MCP retrieval uses `/home/verifier/workspace`.

Daytona tasks explicitly export `/home/agent/workspace` after verification to
`jobs/<run>/<trial>/artifacts/reviewed-workspace/`. Harbor collects remote agent
logs before verification, so the workspace copy made by `test.sh` is not included
in that earlier download. Host validation and bundle resolution accept this
collected export or Docker's `agent/workspace`; they never substitute the
materialized solution when a trial's workspace is missing.
A fresh LLM judgment can still differ from an archived score even with
identical inputs.

`per_task.csv` includes the human and verifier overall scores, computed as the
sum of weights for MET criteria divided by all eligible weights, plus their
difference. Publication scoring requires the complete release; a one-task smoke
uses `evaluate --no-score`.

Use new output and job directories for each run. Existing verdicts, score CSVs
and staging jobs are protected against overwrite. `evaluate --resume` only
reuses a verdict after checking its input identity.

```bash
# RELEASE should be a fresh destination or a verified copy.
python -m adapters.bvb.fetch --with-tools --output-dir "$RELEASE"
python -m adapters.bvb.run_adapter --release-dir "$RELEASE" \
  --task-ids btb-68259d37 --output-dir "$NEW_OUTPUT"
python scripts/smoke_bvb.py --release-dir "$RELEASE" \
  --generated-dir "$NEW_OUTPUT/bvb" --task-id btb-68259d37
# The live judge sends task evidence to its configured model provider.
harbor run --path "$NEW_OUTPUT/bvb/btb-68259d37" \
  --agent oracle --env docker --jobs-dir "$NEW_JOBS" --job-name smoke
```

## Setup

```bash
cd verifier-eval
./scripts/bootstrap.sh                 # all three host verifiers
./scripts/bootstrap.sh archipelago     # or only the ones you need
```

Bootstrap installs each verifier's Python (AAAJ 3.11, AutoRubric 3.12,
Archipelago 3.13) through `uv` and syncs it from its lockfile. The environments
are separate because their Python and LiteLLM constraints conflict. The make
targets below run it for the selected verifiers.

Credentials are loaded from the repository-root `.env`; see
[`.env.example`](../.env.example). AAAJ uses `OPENAI_API_KEY` for its published
`text-embedding-3-large` retrieval and Archipelago `REDUCTO_API_KEY` for
document extraction. The run manifest records key *presence*, never values.

## Stage and evaluate

From the repository root, after `make fetch materialize`:

```bash
export PYTHONPATH=verifier-eval
uv run python -m verifier_runner --release-dir bvb-data/BankerVerifierBench stage --dry-run
uv run python -m verifier_runner --release-dir bvb-data/BankerVerifierBench \
  evaluate --run-id <run-id> --verifier gandalf,aaaj,archipelago,autorubric
```

Gandalf runs inside Harbor during `stage`. AAAJ, Archipelago, and AutoRubric run
afterward on the host against the same staged workspace and trajectory. Partial
smokes must pass `--no-score`; strict score generation requires all 24 tasks.
Generated tasks are read from `datasets/gandalf/bvb/`; staging defaults to
`jobs/staging__bvb/`.

## One-task smoke

`make verifier-smoke` stages one task once and runs Gandalf and the three
host-side verifiers on it. `make gandalf-smoke` runs Gandalf only;
`SMOKE_VERIFIERS` selects any subset:

```bash
make verifier-smoke SMOKE_VERIFIERS=gandalf,archipelago VERIFIER_SMOKE_TASK=btb-68259d37
```

Expected output is under
`verifier-eval/results/<VERIFIER_SMOKE_RUN_ID>/` (fresh timestamped ID by default):

```text
verdicts/{gandalf,aaaj,archipelago,autorubric}/btb-68259d37.json
traces/{gandalf,aaaj,archipelago,autorubric}/btb-68259d37/
traces_atif/{gandalf,aaaj,archipelago,autorubric}/btb-68259d37.json
manifest.json
```

A smoke is an access and configuration check, not an accuracy measurement.

## RewardKit (separate Harbor verifier)

RewardKit is not a host-side runner. Like Gandalf, it runs inside the task
environment and therefore has a separate generated dataset and staging job.
`make rewardkit-claude-smoke` checks it on one task with RewardKit's Claude Code
agent (`ANTHROPIC_API_KEY`). The reported RewardKit comparison instead uses
native **Codex with GPT models, high reasoning effort and a verified Daytona
snapshot**, via `scripts/run_bvb_rewardkit_daytona.py`. The smoke does not
reproduce that configuration.

The RewardKit task image pins `harbor-rewardkit==0.1.2` and Claude Code, uses
the same BVB MCP source, and keeps RewardKit workspace isolation disabled.
Its judge consumes the symlink view as the verifier user rather than using
Gandalf's copy-and-privilege-drop boundary, so the Gandalf smoke does not
certify RewardKit isolation.
Materialization writes `datasets/rewardkit/bvb/`; staging defaults to
`jobs/staging__bvb__rewardkit/`. Its verifier contract is:

```text
verifier/reward.json
verifier/reward-details.json
verifier/judge_trajectories/*.jsonl
```

## Strict scoring

```bash
make analysis RUN_ID=<run-id>
```

The scorer emits pooled UNMET-positive precision, recall, F1, accuracy, MCC,
class rate, and TP/FP/FN/TN only when each task has exactly one complete
canonical verdict. Missing labels, missing tasks, malformed output,
timeouts, duplicate criteria, reordered criteria, or cross-rubric verdicts are
terminal errors.

## Citation

Cite the paper as given in the [repository README](../README.md#citation).

# BankerVerifierBench

Code and paper results for evaluating rubric verifiers against expert human
judgments of agent rollouts on investment-banking tasks. Each task pairs one
reviewed agent rollout (workspace and full trajectory) with a canonical rubric
and adjudicated MET/UNMET labels. A verifier judges the rollout against the
rubric; its decisions are scored against the human labels.

| Resource | Location |
| --- | --- |
| Paper | *Verifying Agents in Rubric-Graded Environments*, NeurIPS 2026 Evaluations and Datasets Track ([citation](#citation)) |
| Dataset | [`handshake-ai-research/BankerVerifierBench`](https://huggingface.co/datasets/handshake-ai-research/BankerVerifierBench), 24 tasks, 2,464 scoring-eligible criteria (831 UNMET) |
| Shared tools | `shared-tools/` in [`handshake-ai-research/bankertoolbench@v1.1`](https://huggingface.co/datasets/handshake-ai-research/bankertoolbench/tree/v1.1) |
| Paper results | [`results/paper/`](results/paper/README.md), every reported measurement on the 24 public tasks |
| Rollout generation | [`results/rollout-generation/`](results/rollout-generation/README.md), tool calls, tokens and cost of the 24 released rollouts |
| Reproduction details | [Reproduction guide](docs/REPRODUCING_BVB.md), [verifier harness](verifier-eval/README.md) |

The code pins dataset commit `e8068910360bcb54459a18c97cfc8d1548da37c1`
(tag `v1.0`) and the SHA-256 of that commit's checksum manifest
in [`adapters/bvb/config.py`](adapters/bvb/config.py). The dataset pins the
BankerToolBench shared-tool revision and archive hashes in
`manifests/shared-tools.json`. Every command below validates both before use.

This is BankerVerifierBench **v1.0**. The upstream BankerToolBench **v1.1**
shared tools have an independent version number.

```bash
git clone https://github.com/Handshake-AI-Research/bankerverifierbench
cd bankerverifierbench
```

## Choose a path

Each level adds one requirement. Levels 0 and 1 and scoring your own verifier
never call a model from this repository.

| Level | Goal | Needs | Commands |
| --- | --- | --- | --- |
| 0 | Check the paper results | Python 3 | `make verify-results` |
| 1 | Fetch and validate the dataset | `uv` | `make fetch validate smoke test` |
| 1 | Evaluate your own verifier | Level 1 and your verifier | `examples/custom_verifier.py`, `bvb score` |
| 2 | Judge one task with the paper's verifiers | Docker, Harbor 0.6.5, provider keys | `make tool-smoke`, `make gandalf-smoke`, `make verifier-smoke` |
| 3 | Reproduce a paper condition | Harbor 0.6.5, Daytona, provider keys | `make daytona-snapshot runtime-snapshots`, `bvb run`, `bvb score` |

Copy [`.env.example`](.env.example) to `.env` for levels 2 and 3; it lists
which key each verifier needs.

### Level 0: check the paper results

```bash
make verify-results
```

This uses only the Python standard library. It checks the bundle's file hashes,
then recomputes every run, condition summary, bootstrap interval, paired
comparison and capability table from the stored criterion predictions. It does
not certify the original native traces or the validity of the human labels.

### Level 1: fetch the dataset and materialize tasks

```bash
uv sync --frozen
make fetch      # dataset into bvb-data/BankerVerifierBench, tools into its shared-tools/
make validate   # hashes, coverage and pinned identities
make smoke      # materialize all 24 tasks and audit the judge inputs
make test       # unit tests and offline scoring checks
```

`make fetch` downloads the pinned dataset tag, then the shared-tool archives
listed in its `manifests/shared-tools.json`, and verifies every hash. It is safe
to rerun. `make smoke` builds Harbor tasks in `datasets/smoke/` and checks that
each exposes exactly the frozen prompt, canonical rubric and authentic selected
rollout. Set `BVB_RELEASE_DIR` to use an existing verified copy.

Downloads need no Hugging Face token. A token only raises download rate limits:
authenticate with `hf auth login`, export `HF_TOKEN`, or set it in the
repository-root `.env` (see [`.env.example`](.env.example)). `make fetch` loads
`.env` automatically. Direct `uv run python -m adapters.bvb.fetch` calls use the
logged-in token or exported `HF_TOKEN`; they do not load `.env`.

Only **10 of the 24 tasks** have input attachments under `task-data/`; the other
**14 tasks have no input files**. All 24 task definitions and selected rollouts
are included. The dataset releases exactly one human-reviewed rollout per task
(12 Gemini 2.5 Pro, 12 GLM-5), with no alternate model candidates.

The public annotations contain the reviewers' decisions and evidence, and the
final adjudicated labels. Internal review packs, workbook references and
annotation-history manifests are not required or distributed.

### Evaluate your own verifier

[`examples/custom_verifier.py`](examples/custom_verifier.py) is a standard-library
template. It reads each task's instruction, canonical rubric and selected
rollout, calls a `judge` function once per criterion and writes predictions as
JSON Lines. Replace `judge` with your verifier, then score:

```bash
uv run python examples/custom_verifier.py --output outputs/my-verifier.jsonl --repeats 3
uv run bvb score --predictions outputs/my-verifier.jsonl
```

Each prediction line holds `task_id`, `evaluation_criterion_index`,
`criterion_text_sha256`, `repeat` and `decision` (`MET` or `UNMET`). `bvb score`
rejects unknown criteria and changed criterion text, and requires each task and
repeat to decide every criterion. It reports the paper's metrics: UNMET-class
precision, recall and F1 pooled over all decisions, accuracy, and weighted-score
MAE. With three repeats on all 24 tasks it adds majority-vote F1 and the
task-bootstrap interval. It prints the paper's main verifiers for reference.
Results on fewer tasks are marked partial and are not comparable to the paper
table. The unmodified template accepts every criterion and reproduces the
always-MET baseline in `results/paper/baselines.json`.

#### Rollout evidence

The task manifest's `label_target_rollout.candidate_path` names each task's
rollout directory, which holds two things:

- **`agent/workspace/`**, the agent's complete final workspace: the task's
  input attachments as the agent received them, its scripts and intermediate
  files, files written by the tools (such as `cik-lookup.json`), and the
  submitted files in `banker_workspace/deliverables/` (Excel, PowerPoint and
  PDF). Judge output quality from these files.
- **`agent/trajectory.json`**, the full trajectory in ATIF v1.6. `steps` holds
  the task prompt (`source: user`) and every agent turn, with `message`,
  `reasoning_content`, `tool_calls` (tool name and arguments) and `observation`
  (tool outputs, including the tool server's responses). Use it for criteria
  about the process, and to see which source data the agent retrieved.

`task-data/<task-id>/Inputs/` holds the original attachments of the 10 tasks
that have any. Byte-identical copies are already in the workspace, so it is
needed only to tell inputs apart from the agent's own files.

Criteria that check figures against SEC filings, market data or logos can be
verified independently only through the tool server: `tool-profiles/mcp-server/`
in the release, with the shared-tool archives and
`manifests/tool-archive-index.json`. It reads the SEC archives with the `zstd`
command. In the paper, Gandalf and RewardKit had the tool server inside the task
container (Level 2 below); AAAJ, Archipelago and AutoRubric judged from the
workspace and trajectory alone. Their integrations in
[`verifier-eval/verifier_runner/runners/`](verifier-eval/verifier_runner/runners/)
show how each one passes this evidence to its model.

### Level 2: judge one task locally

```bash
uv tool install harbor==0.6.5          # the version every measured run used
make tool-smoke                         # build the local images, read real tool archives; no model calls
make gandalf-smoke                      # Gandalf in Docker via Harbor; GEMINI_API_KEY
make verifier-smoke                     # Gandalf, AAAJ, Archipelago and AutoRubric
```

`tool-smoke` builds the local Gandalf and RewardKit images and, inside each,
calls the configured MCP server over stdio as the judge user (UID 10003).
It retrieves a logo, a market-data workbook and SEC data through the setuid
wrapper, then reads and hash-checks every export as the judge. This catches
missing decompressors and unreadable exports before any paid run.
Local images apply the same group-readable logo-export correction as the
derived Daytona runtimes; `bvb-task-manifest.json` records the correction and
its source hashes. The frozen dataset and recorded Daytona base runtime are
unchanged. Regenerate existing local tasks to include the correction.
`gandalf-smoke` and `verifier-smoke` stage task `btb-68259d37` under Harbor.
Gandalf judges it inside the task container, with the tool server available. `verifier-smoke` installs the host
verifiers' environments on first use, then runs them on the same staged
evidence. Select verifiers with `SMOKE_VERIFIERS=gandalf,archipelago`. The
optional `make rewardkit-claude-smoke` checks RewardKit with its Claude Code
agent; the paper's RewardKit conditions use Codex on Daytona instead.

### Level 3: reproduce a paper condition on Daytona

```bash
make daytona-snapshot runtime-snapshots   # base and derived snapshots, built once
uv run bvb list                           # every measured condition, its coverage and F1
uv run bvb show M05                       # recorded settings, results, cost and command
uv run bvb run --condition M05 --output outputs/m05
uv run bvb score --verdicts outputs/m05
```

[`configs/paper/conditions.toml`](configs/paper/conditions.toml) records the
runner and native settings of each measured condition: model comparisons
(M01–M20), exploratory screens (E01–E03) and the batching study (B-*). `bvb run`
invokes that runner unchanged. Add `--tasks` and `--repeats 1` for a cheap
first run, or `--dry-run` to print the command. Host-verifier conditions (AAAJ,
Archipelago, AutoRubric) judge the evidence staged by an M01 run and need
`--evidence-run outputs/m01`. M01 itself is scored with
`scripts/score_bvb_verifier_against_humans.py`; see the
[reproduction guide](docs/REPRODUCING_BVB.md).

Recorded API cost was about $3,900 across all 29 conditions. One Gandalf task
run with Gemini 3 Flash averaged $0.41, and a full condition ranges from $13 to
$970. The guide lists disk, sandbox, time and cost requirements per condition.

## Evaluation contract

- **Prompt:** `task-source/<task-id>/instruction.md`, unchanged.
- **Criteria:** `manifests/evaluation-rubrics/`, with the reviewed wording,
  weights, order and exclusions. The historical `tests/rubric.json` in task
  sources is not a scoring rubric.
- **Evidence:** only `agent/workspace/` and `agent/trajectory.json` inside the
  selected rollout directory, matched by hash against the release manifests. Pass
  these two paths to your verifier alongside the prompt and canonical criteria.
  Rollout directories contain only this evidence; earlier grader outputs, rewards,
  generation logs and duplicate deliverables are omitted. Native verifiers access
  the evidence through their own interfaces.
- **Truth:** adjudicated labels in `human-adjudication/canonical/`, used only
  for scoring and never shown to a judge.
- **Metrics:** UNMET is the positive class. The harness reports defect
  precision, recall and F1, accuracy, weighted-score MAE, cost and time. Pooled
  single-run decisions and three-run majority votes are reported separately.
  Headline metrics require complete task coverage.

Generated tasks carry an input contract with the release identity. The harness
rejects changed identities, malformed criteria, missing criteria and missing tasks.
Native errors, abstentions, retries and unknown usage remain visible in outputs.

## Results

[`results/paper/`](results/paper/README.md) holds 1,749 saved task/repeat
results across 29 conditions on the 24 public tasks: model comparisons, batching
ablations and exploratory screens, plus the nine-capability analysis.
`bvb list` shows which conditions have complete coverage. The bundle's README
lists what the paper does not report and what is not included.

Where each table and figure of the paper comes from (condition IDs are rows of
[`results/paper/conditions.csv`](results/paper/conditions.csv)):

| Paper | Source |
| --- | --- |
| Table 1, capabilities | `capabilities/tab_taxonomy_rows.tex`, `capabilities/table1_capability_distribution.csv` |
| Figure 1, F1 and cost | M01–M08 (Gandalf), M13–M15 (Archipelago), M16–M18 (AAAJ), M19 (AutoRubric) |
| Table 2a, design principles | Not released |
| Table 2b, batch size | B-b4, B-b8, B-b16, B-b32, B-b64, B-split2 |
| Table 2c, model | M01–M08 |
| Table 3, OpenClaw | Not released |
| Table 4, per-capability error rates | `capabilities/tab_combined_rows.tex`, `capabilities/bvb_primary_error_rates.csv`, `capabilities/opc_primary_error_rates.csv` |
| Tables A1 and A2, rollout generation and verifier cost | [`results/rollout-generation/`](results/rollout-generation/README.md); verifier costs from M01, M13, M16, M19 |
| Table A3, Gandalf configuration | [`configs/paper/conditions.toml`](configs/paper/conditions.toml) (BVB column) |
| Table A4, RewardKit matched models | M09–M12 against M05–M08 |
| Table A5, RewardKit inputs | M10 (the marked row); the other input conditions are not released |
| Table A6, confusion counts | BVB columns from M01, M13, M16, M19; the comparisons in its text also use M05, M08, M14; OpenClaw columns not released |

All paths in the second column are under `results/paper/` unless linked.

## Repository map

| Path | Contents |
| --- | --- |
| `adapters/bvb/` | Release fetch and validation, Harbor task generation, Daytona snapshots, the `bvb` command |
| `configs/paper/` | Runner and settings of every measured condition |
| `examples/` | Template for evaluating your own verifier |
| `verifier-eval/` | Native verifier runners, staging and strict scoring |
| `scripts/` | Condition runners, scoring and the offline results check |
| `results/paper/` | Hash-manifested paper results |
| `tests/` | Unit tests (`make test`) |

Downloaded data (`bvb-data/`), generated tasks (`datasets/`), run outputs
(`jobs/`, `outputs/`) and credentials (`.env`) are not tracked.

## Citation

```bibtex
@inproceedings{duecker2026verifying,
  title     = {Verifying Agents in Rubric-Graded Environments},
  author    = {Markus D{\"u}cker and Vaibhav Kumar and Yi Liu and Ronak Chaudhary and Andreas Plesner and Francisco Guzm{\'a}n and Anish Athalye},
  booktitle = {The Fortieth Annual Conference on Neural Information Processing Systems Evaluations and Datasets Track},
  year      = {2026},
  note      = {Accepted for publication}
}
```

## License

Code is Apache-2.0; see [LICENSE](LICENSE). Vendored and adapted third-party
code is listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The dataset
and the BankerToolBench tools are distributed under their own terms on Hugging Face.

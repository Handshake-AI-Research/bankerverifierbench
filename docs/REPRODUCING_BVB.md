# Reproducing BankerVerifierBench results

The code pins BankerVerifierBench commit `e8068910360bcb54459a18c97cfc8d1548da37c1`
(tag `v1.0`), and the SHA-256 of its checksum
manifest, in `adapters/bvb/config.py`: 24 tasks and 2,464 scoring-eligible
criteria. The [root README](../README.md) introduces the four levels; this guide
covers what each step checks and how the measured conditions map to commands.

## Requirements and budget

| Level | Host | Storage and compute | Paid services |
| --- | --- | --- | --- |
| 0 | Python 3 | None | None |
| 1 | Python 3.12+, `uv` | 5.8 GB for the release (5.2 GB of it shared tools), 0.25 GB for generated tasks | None |
| 2 | Docker, Harbor 0.6.5 | About 6 GB per local task image | Model APIs; Reducto for Archipelago |
| 3 | Daytona account | Each sandbox: 4 vCPU, 8 GB memory, 20 GB disk | Model APIs, Daytona |

Install Harbor with `uv tool install harbor==0.6.5`; every measured run recorded
that version. The measured runs were launched from macOS. Daytona sandboxes
inherit the resources of the snapshot they start from. Each Daytona condition
records how many task sandboxes it ran at once, from 5 to 25; `bvb show` prints
it as `concurrency`, and your Daytona account limits must allow it.

`bvb show <id>` prints a condition's recorded API cost, and
`results/paper/conditions.csv` adds the mean seconds per task run. Recorded API
cost was about $3,900 across all 29 conditions, $1,960 of it for M01–M20. A full
condition ranges from $13 (M11, three repeats) to $970 (E03), and one Gandalf
task run with Gemini 3 Flash (M01) averaged $0.41. Mean time per task run ranges
from about 1 minute (Archipelago) to 35 minutes (B-b4). These are recorded
estimates, not invoices; they exclude Daytona, Reducto and other infrastructure
charges.

## 1. Check the paper results

```bash
make verify-results
```

Python standard library only. This checks hashes, then recomputes every run's
confusion counts and weighted scores from the bundled predictions. From those it
recomputes the condition summaries, bootstrap intervals, paired comparisons,
batching sensitivities and capability tables. It does not launch graders, need
secrets or re-adjudicate labels. See
[`results/paper/README.md`](../results/paper/README.md).

## 2. Fetch the pinned inputs and check materialization

```bash
uv sync --frozen
make fetch validate smoke test
```

`make fetch` downloads the pinned dataset revision into
`bvb-data/BankerVerifierBench/`. It then downloads the shared-tool archives named
in the release's `manifests/shared-tools.json` from
`handshake-ai-research/bankertoolbench` at the recorded revision. Every file is
hash-checked against `manifests/files.sha256` and `shared-tools.json`.
Validation also checks the release name and schema, counts, the MCP profile
digest and the tool-archive index. Set `BVB_RELEASE_DIR` to use a verified copy
elsewhere.

`make smoke` materializes all 24 tasks into `datasets/smoke/bvb/` and audits
each. To inspect another build, run
`scripts/smoke_bvb.py --release-dir <release> --generated-dir <build>/bvb`.

The task prompt is frozen `task-source/<task>/instruction.md`. Criteria come from
`manifests/evaluation-rubrics/`, **not** historical `task-source/*/tests/rubric.json`
or rollout `verifier/info.json`. The selected rollout mapping controls workspace
and full trajectory. Canonical labels are scoring-only inputs. See
[`verifier-eval/README.md`](../verifier-eval/README.md) for filesystem paths,
native evidence interfaces and score outputs.

## 3. Runtimes

Every Daytona snapshot is named by its inputs, and runners verify the recorded
identity inside a sandbox before any model call.

| Snapshot | Build | Used by |
| --- | --- | --- |
| Base `bvb-r<rev>-m<mcp>-i<index>-a<archives>-t<runtime>` | `make daytona-snapshot` | M01 |
| Derived `<base>-llm-<id>` | `make runtime-snapshots` | Gandalf model comparisons, M02–M08 |
| Derived `<base>-rewardkit-<id>` | `make runtime-snapshots` | RewardKit comparisons, M09–M12 |

The base snapshot runs the pinned native Gandalf 1.0.0 stack. The derived
Gandalf runtime installs LiteLLM 1.104.0 for newer model IDs. It also backports
the upstream OpenHands reasoning-capability check into SDK 1.14.0 and makes
exported logos group-readable. The derived RewardKit runtime adds RewardKit
0.1.2, Codex 0.160.1 behind `codex-wrapper.py`, and the same logo fix. The
patches are in `adapters/bvb/template/_environment-*/`. Each is checked against
the original and patched file hashes recorded for the measured runs.
`adapters/bvb/runtime_snapshots.py` reproduces the measured runtimes' identity
records exactly; only the base snapshot name changes with the dataset revision.
RewardKit's transitive dependencies resolve at build time.

## 4. Run a measured condition

`configs/paper/conditions.toml` records the runner and native settings
of every condition in `results/paper/`. The `bvb` command (installed by `uv sync`)
turns an entry into the runner's own command line without changing the runner:

```bash
uv run bvb list
uv run bvb show M05      # settings, measured results and run command
uv run bvb run --condition M01 --output outputs/m01 --tasks btb-68259d37 --repeats 1
uv run bvb run --condition M05 --output outputs/m05 --tasks btb-68259d37 --repeats 1
uv run bvb run --condition M19 --output outputs/m19 --tasks btb-68259d37 --repeats 1 \
  --evidence-run outputs/m01
uv run bvb run --condition B-b16 --output outputs/b16 --tasks btb-68259d37 --repeats 1
```

Add `--dry-run` to print the command only. Omit `--tasks` and `--repeats` for
the full condition. `--output` must be a fresh directory inside the repository,
such as `outputs/<id>`, because Harbor stages tasks by repository-relative path.
Host-verifier conditions (AAAJ, Archipelago, AutoRubric) read the evidence
staged by an M01 run, as in the paper. Each entry was checked against the
native settings recorded for every bundled run: model, reasoning effort, batch
partition, timeouts, runtime snapshot, LiteLLM version and concurrency. Where
recovery launches used different settings, the entry records those of the
majority of runs and its `notes` describe the rest.

| Runner | Conditions | Script |
| --- | --- | --- |
| `gandalf-daytona` | M01 | `scripts/run_bvb_gandalf_daytona.py` |
| `gandalf-matrix` | M02–M08 | `scripts/run_bvb_gandalf_matrix.py` |
| `rewardkit-daytona` | M09–M12 | `scripts/run_bvb_rewardkit_daytona.py` |
| `host-matrix` | M13–M20, E01–E03 | `scripts/run_bvb_verifier_matrix.py` |
| `gandalf-batching` | B-b4 … B-split2 | `scripts/run_bvb_gandalf_batching.py` |

Gandalf 1.0.0 can only split a rubric into N near-equal batches. The batching
study (B-b4 to B-b64) fixes the number of criteria per judge session instead, so
`adapters/bvb/batching.py` rebuilds the measured overlay of Gandalf 1.0.0. It
downloads the hash-pinned PyPI wheel, adds an exact-size `batch_size` mode and
hooks that record each session's actual input, then checks every file against
the measured overlay hashes. The judge prompt, templates and verdict parsing
are unchanged. B-split2 uses the native `batch_splits = 2` with the same
overlay, as a matched control. Every condition runs Gemini 3 Flash at high
effort on the derived gandalf-llm runtime, with two judge sessions at a time
per task, a 1,740-second session cap and a 6-hour verifier limit.

The runner accepts a run only if every judge session received exactly its
ordered share of the canonical rubric together with the reviewed evidence, and
if token, cost and score accounting reconcile. These are the measured study's
acceptance checks; replaying them on its archived trials reproduces its audits.
The study retried an invalid task and repeat at most twice; to do the same,
rerun failed tasks into a new output directory with `--tasks` and
`--repeat-start`. Task sets generated by this code match the measured ones
byte for byte, apart from dataset identity, snapshot name and local paths.

Live keys belong in an untracked `.env`; `bvb run` loads it. Copy
[`.env.example`](../.env.example), which lists the keys each verifier needs:
`DAYTONA_API_KEY` for remote sandboxes and `GEMINI_API_KEY` or `OPENAI_API_KEY`
for the selected model. AAAJ also needs `OPENAI_API_KEY` for its native
embeddings and Archipelago `REDUCTO_API_KEY` for document extraction.

### Score a run

Every runner except M01's writes one Verdict file per task and repeat below
`<output>/…/repeat-<N>/`. Score them against the human labels with the paper's
metrics:

```bash
uv run bvb score --verdicts outputs/m05
uv run bvb score --verdicts outputs/b16 --json
```

Point `--verdicts` at one verifier and model; a host-matrix output directory
holds one subdirectory per verifier and model. M01 leaves a Harbor job, which
`scripts/score_bvb_verifier_against_humans.py` validates and scores:

```bash
uv run python scripts/score_bvb_verifier_against_humans.py --job-dir jobs/m01 --output outputs/m01/score.json
```

Runs on fewer than 24 tasks, or with incomplete repeats, are reported as
partial and are not comparable to `results/paper/conditions.csv`.

Several conditions were completed with recovery launches after quota or
infrastructure failures; the registry `notes` say where. The bundle's attempt
records cover the public tasks. A fresh run is a new measurement:
provider-side model behavior can differ from the measured runs even with
identical inputs and runtimes.

## 5. Protect the measurement contract

Use fresh run IDs and output directories. Preserve failed attempts and reuse only
identity-validated accepted slots. `UNMET` is positive: rejecting human-MET is a
false positive; accepting human-UNMET is a false negative. Overall scores weight
MET criteria by rubric weight; criterion F1 uses criterion counts. Report both.

Pool all fixed repeats for individual-run metrics. Majority voting requires all
three judgments for every criterion. A one-repeat screen is fixed repetition 1,
never the best of several attempts. Bootstrap tasks as clusters; do not treat
repeated criteria as independent benchmark tasks.

Development tests run only when invoked, for example `make test`; they make no
judging calls. Runtime input and trace audits validate actual configuration
and completeness. They must not replace performance mistakes with favorable
reruns. Reader/MCP availability and successful retrieval are separate findings.

Any runtime fix must be a separately named condition. Preserve the original
attempts, and establish whether the fix changes the scientific comparison.

## Citation

Cite the paper as given in the [repository README](../README.md#citation).

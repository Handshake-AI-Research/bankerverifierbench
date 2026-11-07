# Paper results: BankerVerifierBench, 24 tasks

Every verifier measurement and capability analysis reported for BankerVerifierBench,
on the public release
[`handshake-ai-research/BankerVerifierBench`](https://huggingface.co/datasets/handshake-ai-research/BankerVerifierBench)
at tag `v1.0`: **24 tasks and 2,464 criteria
(831 UNMET, 1,633 MET)**. UNMET is the positive class throughout.

`manifest.json` and `experiment-data.json.gz` record dataset commit
`e8068910360bcb54459a18c97cfc8d1548da37c1` (tag `v1.0`), the commit the code pins.

Verify everything offline, with no credentials, dependencies or model calls:

```bash
make verify-results        # python3 scripts/bvb_public_results.py check
```

The check confirms every file hash, then recomputes every run's confusion counts and
weighted scores from the criterion predictions. From those it recomputes every condition
summary, task-bootstrap interval, paired comparison, batching sensitivity and capability
table, and requires them to equal the files here.

## Contents

| File | What it holds |
|---|---|
| `experiment-data.json.gz` | 1,749 saved task/repeat runs across 29 conditions, all 179,455 criterion predictions, the canonical rubric, batching sessions and attempt records. |
| `conditions.csv` | One row per condition: coverage, UNMET F1, precision, recall, accuracy, overall MAE, majority-vote F1, API cost and time. |
| `paired-comparisons.json` | Paired task-bootstrap intervals for the fixed-repeat model screens and the batching study. |
| `batching-sensitivities.json` | Flagged-slot, source-artifact and native-retry sensitivities for the batching study. |
| `baselines.json` | Constant all-UNMET and all-MET reference levels. |
| `capabilities/` | Capability analysis under the paper's nine capabilities (Table 1, the combined error table and its appendix, the Archipelago direction analysis, the paired differences and the replicate-tag sensitivity), plus the frozen tags it uses and the OpenClaw rates and intervals of the combined table. |

In `experiment-data.json.gz`, read `run_keys` for run columns. Criterion rows are
`[condition, task, evaluation_index, human, repeat1, repeat2, repeat3]`, with `null` for a
missing repeat. Rubric rows are `[task, evaluation_index, source_index, category, weight,
human, canonical_text, text_sha256]`. Null usage means unknown, not zero. Every run's
selected rollout and rubric hash match the public release.

## Headline results

Pooled individual-run metrics over three repeats (72 runs) unless a condition has one
repeat (24 runs). The interval is the task-bootstrap 95% interval for UNMET F1. Overall
MAE is the mean absolute difference between weighted verifier and human scores, in points
out of 100. API cost is a recorded estimate, not an invoice; it excludes Daytona, Reducto
and other infrastructure charges.

| ID | Verifier | Model / reasoning | Runs | UNMET F1 [95% CI] | Majority F1 | Accuracy | MAE | USD/run |
|---|---|---|---:|---|---:|---:|---:|---:|
| M09 | RewardKit | GPT-5.4 / high | 72 | 0.880 [0.852, 0.902] | 0.882 | 91.2% | 5.14 | 1.45 |
| M01 | Gandalf | Gemini 3 Flash / high | 72 | 0.864 [0.835, 0.886] | 0.887 | 90.8% | 3.33 | 0.41 |
| M13 | Archipelago | Gemini 3 Flash / default | 72 | 0.852 [0.812, 0.882] | 0.851 | 89.7% | 3.64 | 1.95 |
| M19 | AutoRubric | Gemini 3 Flash / default | 72 | 0.833 [0.792, 0.866] | 0.831 | 88.9% | 5.16 | 3.37 |
| M16 | AAAJ | Gemini 3 Flash / default | 72 | 0.702 [0.638, 0.751] | 0.721 | 81.5% | 9.40 | 2.22 |
| M05 | Gandalf | GPT-5.4 / high | 72 | 0.892 [0.867, 0.911] | 0.897 | 92.2% | 3.80 | 2.06 |
| M06 | Gandalf | GPT-5.4 mini / high | 72 | 0.883 [0.854, 0.905] | 0.892 | 92.0% | 3.28 | 0.58 |
| M10 | RewardKit | GPT-5.4 mini / high | 72 | 0.884 [0.857, 0.905] | 0.894 | 92.1% | 3.01 | 0.44 |
| M04 | Gandalf | Gemini 3.8 Flash / high | 24 | 0.895 | — | 92.6% | 4.90 | 2.03 |

The constant references are all-UNMET F1 0.504 and all-MET accuracy 66.3% (F1 0). All 29
conditions, including the batching study and the partial exploratory screens, are in
`conditions.csv`. AutoRubric with GPT-5.4 (M20) was held on cost after one public task and
is not a result.

Batching (Gandalf, Gemini 3 Flash): b16 has the highest pooled UNMET F1 (0.880) of the six
partition settings. Its paired task-bootstrap interval against split2 (0.862) and b64 (0.858)
excludes zero; against b4, b8 and b32 it does not. Comparisons are exploratory and not
multiplicity-adjusted. The flagged-slot and source-artifact sensitivities keep b16 highest.

Capabilities (primary tag, three repeats pooled, error rate in %): Gandalf has the lowest
point estimate among the four Gemini 3 Flash verifiers in 7 of 9 capabilities. Archipelago
is lowest on governance narrative, AutoRubric on cross-component linkage. RewardKit uses
GPT-5.4, so it is not model-matched. The paired intervals in
`capabilities/paired_error_differences_vs_gandalf.csv` resolve fewer of these differences
than the point estimates suggest. See `capabilities/summary.json`.

On OpenClaw, aggregate error is 5.3% for Gandalf, 21.8% for Archipelago, 19.8% for AAAJ and
7.6% for AutoRubric. Gandalf is lowest in 5 of the 6 capabilities with at least five
criteria; AutoRubric is lowest on document labeling. Against AutoRubric no capability's
difference is resolved (`capabilities/opc_paired_error_differences_vs_gandalf.csv`).

## OpenClaw aggregates

The OpenClaw (OPC) rows in `capabilities/opc_primary_error_rates.csv` and their paired
intervals in `capabilities/opc_paired_error_differences_vs_gandalf.csv` come from a separate
benchmark: 396 criteria over 24 episodes, tagged with the same nine capabilities. They cover
the four verifiers on Gemini 3 Flash, three passes each, pooled. The intervals are
task-clustered paired bootstraps over the 24 episodes (seed 20261102, 2,000 draws). OpenClaw
data are not public, so only these aggregates are included; the check verifies the rates and
intervals against each other, and `manifest.json` records the hashes of the verdict exports
they were computed from. With fewer than five criteria a capability's OPC row is shown as "-".

## Not reported in the paper

The bundle also holds analyses the paper does not report:

- the exploratory screens E01–E03 and AutoRubric with GPT-5.4 (M20, stopped after one task);
- the paired comparisons in `paired-comparisons.json` and the sensitivities in
  `batching-sensitivities.json`;
- the paired capability intervals (`capabilities/paired_error_differences_vs_gandalf.csv`,
  `capabilities/opc_paired_error_differences_vs_gandalf.csv`);
- the Archipelago direction analysis (`capabilities/archi_bimodal_rows.tex`,
  `capabilities/archipelago_fp_fn.csv`), the appendix variant of the combined table
  (`capabilities/tab_combined_appendix_bvb_rows.tex`) and the replicate-tag sensitivity
  (`capabilities/bvb_primary_error_rates_replicate_tags.csv`).

The check does not produce the paper's appendix paired intervals: the error-direction
comparisons (10,000 draws, seed 20261106) and the matched RewardKit comparison (10,000
draws, seed 20261103). Both are task-clustered paired bootstraps that can be computed from
the criterion predictions in `experiment-data.json.gz`.

Not included: the Gandalf design-principle study, the RewardKit input-condition runs, the
OpenClaw criterion-level data, the taxonomy model request/response caches, the capability
adjudication sheets, and native verifier traces.

## Citation

Cite the paper as given in the [repository README](../../README.md#citation).

## License

The results are released with the code under Apache-2.0. Criterion text and labels come
from the dataset and remain under the dataset's own license.

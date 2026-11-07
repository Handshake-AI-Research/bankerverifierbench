# Rollout generation: BankerVerifierBench, 24 rollouts

Usage of the 24 released agent rollouts, one per task (12 Gemini 2.5 Pro, 12 GLM-5,
OpenCode 1.17.9 via OpenRouter), read from each rollout's `agent/trajectory.json` in
the public release at tag `v1.0`. These are the rollouts the human labels refer to;
the statistics describe generating them, not generating the dataset as a whole.

Verify offline, with no credentials or downloads (also part of `make verify-results`):

```bash
python3 scripts/bvb_rollout_generation.py check
```

With `--release <dir>` the check also re-extracts every row from the downloaded
release trajectories and requires identical bytes.

| File | What it holds |
|---|---|
| `rollouts.csv` | One row per rollout: task, model and OpenRouter route, harness, trajectory hash, steps, tool calls, prompt, cached and completion tokens, recorded API cost. |
| `summary.json` | Per model and overall: tool calls (mean, median, range), mean steps, token totals, cache share, cost (total, mean, median, most expensive rollout and its share). Also each complete verifier condition's cost per dataset pass and its ratio to the generation cost. |
| `manifest.json` | Dataset revision and checksum-manifest hash, the hash of `results/paper/conditions.csv` used for the ratios, and hashes of the files above. |

Cost is the recorded model API cost in each trajectory's `final_metrics`, as reported
via OpenRouter and reflecting prompt-caching discounts; it is an estimate, not an
invoice. Tool calls are the calls recorded in the trajectory steps.

| Model | Rollouts | Tool calls, mean (range) | Cache share | Cost, total | Mean | Median |
|---|---:|---|---:|---:|---:|---:|
| Gemini 2.5 Pro | 12 | 21.1 (3–44) | 73.5% | $11.50 | $0.96 | $0.50 |
| GLM-5 | 12 | 29.9 (11–52) | 88.2% | $2.86 | $0.24 | $0.17 |
| All | 24 | 25.5 (3–52), median 24 | 78.6% | $14.36 | $0.60 | $0.29 |

One rollout (`btb-1b181d77`, Gemini 2.5 Pro, 15.1M prompt tokens) accounts for $4.72,
a third of the total.

The verification ratios compare one verifier pass over the 24 tasks with generating
the 24 rollouts. They depend on the verifier configuration (from 0.3× to 10.1× across
complete conditions; with Gemini 3 Flash, 0.7× for Gandalf to 5.6× for AutoRubric)
and on the models' list prices, which differ between the generation and verifier
models. They are specific to these rollouts, which are shorter and cheaper than many
agent runs, and should not be read as a general cost relation between generation and
verification.

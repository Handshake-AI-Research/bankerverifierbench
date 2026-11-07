# Third-party notices

This repository's own code is licensed under Apache License 2.0 (see `LICENSE`).
It also contains the following third-party material, each under its own license.
Local modifications are recorded next to each copy.

## Vendored source

| Path | Upstream | Version | License |
| --- | --- | --- | --- |
| `verifier-eval/verifiers/archipelago/grading/` | [Mercor-Intelligence/archipelago](https://github.com/Mercor-Intelligence/archipelago), `grading/` subtree only | commit `3f4a8234a27d71a17e0aa19e60f116440bcd1481` | Apache License 2.0, `verifier-eval/verifiers/archipelago/LICENSE` |
| `verifier-eval/verifiers/aaaj/agent_as_a_judge/` | [metauto-ai/agent-as-a-judge](https://github.com/metauto-ai/agent-as-a-judge) | commit `43c2fa7b84ef0c1522e8250576e6798bace29dd0` | MIT License, Copyright (c) 2024 metauto.ai, `verifier-eval/verifiers/aaaj/LICENSE` |

Both copies carry research adaptations for BVB. They are described in each
directory's `UPSTREAM.md` and captured as `local-modifications.patch`. Results
produced with them are adapted evaluations, not unmodified upstream reproductions.

## Adapted files

The batching study's exact-batch-size integration in `adapters/bvb/batching.py`
is applied to **gandalf-the-grader 1.0.0**, published by
[Handshake AI Research](https://github.com/Handshake-AI-Research/gandalf-the-grader)
under Apache License 2.0. The wheel is downloaded from PyPI and its original
package files are verified against `adapters/bvb/batching_runtime/native-files.json`
when tasks are built; the integration does not replace the native judge with a
separate grading implementation. The modified files are not redistributed; their
expected hashes are in `adapters/bvb/batching_runtime/overlay-files.json`.

The Harbor task template in `adapters/bvb/template/` is adapted from the
BankerToolBench Harbor adapter by Handshake AI Research.

## Installed dependencies

Pinned packages such as `autorubric==1.0.1` (`verifier-eval/verifiers/autorubric/`),
gandalf-the-grader, OpenHands, Harbor and harbor-rewardkit are installed from
their package indexes and are not redistributed here. Their lock files pin
versions; their own licenses apply.

## Data

The dataset is distributed separately on Hugging Face
([`handshake-ai-research/BankerVerifierBench`](https://huggingface.co/datasets/handshake-ai-research/BankerVerifierBench))
under its own terms. It includes the BankerToolBench MCP server source under
`tool-profiles/mcp-server/`. The shared-tool archives are downloaded from
[`handshake-ai-research/bankertoolbench`](https://huggingface.co/datasets/handshake-ai-research/bankertoolbench)
and are not part of either repository's code.

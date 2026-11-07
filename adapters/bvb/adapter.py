"""Manifest-driven BankerVerifierBench Harbor task materialization."""

from __future__ import annotations

import hashlib
import json
import tomllib
import re
import shlex
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomli_w
import yaml

from .contract import REPLAY_EVIDENCE_GUIDANCE, REPLAY_VIEW, canonical_rows, input_identity, validate_rubric
from .release import BVBRelease, ReleaseError, sha256_file, tree_digest
from .schema import BVBTask

TEMPLATE_DIR = (
    Path(__file__).resolve().parent
    / "template"
)
_RUNTIME_TEMPLATE_FILES = (
    Path("environment/Dockerfile"),
    Path("environment/agent-python-requirements.txt"),
)

_PROVIDER_API_KEY_ENV_VARS = {
    "anthropic/": "ANTHROPIC_API_KEY",
    "openai/": "OPENAI_API_KEY",
    "gemini/": "GEMINI_API_KEY",
    "openrouter/": "OPENROUTER_API_KEY",
    "vertex_ai/": "GOOGLE_API_KEY",
}
DEFAULT_VERIFIER_MODEL = "gemini/gemini-3-flash-preview"
DEFAULT_REASONING_EFFORT = "high"
DEFAULT_EXTENDED_THINKING_BUDGET = 200_000
REASONING_EFFORTS = ("low", "medium", "high", "xhigh", "none")
DEFAULT_REWARDKIT_VERIFIER_MODEL = "anthropic/claude-sonnet-4-6"
DEFAULT_REWARDKIT_AGENT = "claude-code"
DEFAULT_REWARDKIT_VERSION = "0.1.2"
DEFAULT_REWARDKIT_JUDGE_TIMEOUT_SEC = 1_500
DEFAULT_REWARDKIT_HARBOR_TIMEOUT_SEC = 1_800.0


def effective_reasoning_config(
    reasoning_effort: str | None = DEFAULT_REASONING_EFFORT,
    extended_thinking_budget: int | None = DEFAULT_EXTENDED_THINKING_BUDGET,
    reasoning_max_tokens: int | None = None,
) -> dict[str, Any]:
    return {
        "reasoning_effort": reasoning_effort,
        "extended_thinking_budget": extended_thinking_budget,
        "reasoning_max_tokens": reasoning_max_tokens,
        "source": (
            "BVB generated gandalf-the-grader-judge wrapper over "
            "OpenHands SDK 1.14.0"
        ),
    }


EFFECTIVE_REASONING_CONFIG = effective_reasoning_config()
_DAYTONA_MCP_ENV = (
    "BVB_TOOL_ARCHIVE_INDEX=/opt/mcp-server/data/tool-archive-index.json",
    "BVB_TOOL_ARCHIVE_ROOT=/opt/mcp-server/data",
    "BVB_TOOL_CACHE_ROOT=/var/cache/bvb-tools",
    "BVB_TOOL_CACHE_MAX_BYTES=2147483648",
)


def runtime_template_sha256() -> str:
    """Hash snapshot-affecting runtime files so stale images cannot be reused."""
    digest = hashlib.sha256()
    for relative_path in _RUNTIME_TEMPLATE_FILES:
        path = TEMPLATE_DIR / relative_path
        digest.update(relative_path.as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\n")
    return digest.hexdigest()


def snapshot_revision_label(release_revision: str) -> str:
    """Short revision for snapshot names: a commit prefix, or a tag such as v1.0 as v1-0."""
    return re.sub(r"[^A-Za-z0-9]", "-", release_revision[:8])


def _mcp_server_config(execution_environment: str, *, lazy_tools: bool = False) -> dict[str, Any]:
    config: dict[str, Any] = {
        "name": "mcp-server",
        "transport": "stdio",
        "command": "/usr/bin/mcp-server",
    }
    if execution_environment == "daytona" or lazy_tools:
        # Gandalf forwards only a small environment allowlist to its inner
        # OpenHands process. Launch through env so lazy-tool configuration
        # reaches the MCP subprocess even when BVB_TOOL_* was stripped.
        config["command"] = "/usr/bin/env"
        config["args"] = [*_DAYTONA_MCP_ENV, "/usr/bin/mcp-server"]
    return config


@dataclass(frozen=True)
class MaterializedTask:
    """Generated task metadata used by smoke checks and downstream staging."""

    task_id: str
    task_dir: Path
    trajectory_provenance: str
    workspace_provenance: str
    generated_manifest: Path


def _api_key_env_var(model: str) -> str:
    for prefix, env_var in _PROVIDER_API_KEY_ENV_VARS.items():
        if model.startswith(prefix):
            return env_var
    raise ReleaseError(f"Unsupported verifier model provider: {model}")


def _copytree(source: Path, destination: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, destination, dirs_exist_ok=True)


def _find_authentic_agent_dir(
    task: BVBTask,
    candidate_root: Path,
    artifact_root: Path | None,
) -> Path | None:
    """Resolve real artifacts from the release or an optional imported archive."""
    locations = [candidate_root / "agent"]
    if artifact_root is not None and task.selected_rollout is not None:
        selected = task.selected_rollout
        locations.extend(
            [
                artifact_root / task.task_id / selected.model / "agent",
                artifact_root / selected.model / selected.trial_name / "agent",
                artifact_root / selected.trial_name / "agent",
            ]
        )
    for location in locations:
        if (location / "trajectory.json").is_file():
            return location
    return None


def _declared_replay_hashes(task: BVBTask) -> tuple[str | None, str | None]:
    """Read future package replay hashes without accepting undeclared artifacts."""
    replay = task.manifest.get("replay_artifacts") or {}
    workspace = replay.get("workspace") or {}
    trajectory = replay.get("trajectory") or {}
    return (
        replay.get("workspace_tree_sha256") or workspace.get("tree_sha256"),
        replay.get("trajectory_sha256") or trajectory.get("sha256"),
    )


class BVBAdapter:
    """Generate BVB v1.0 Harbor tasks exclusively from release artifacts."""

    verifier_kind = "gandalf"

    def __init__(
        self,
        release: BVBRelease,
        *,
        output_dir: Path,
        tool_root: Path,
        verifier_model: str = DEFAULT_VERIFIER_MODEL,
        reasoning_effort: str | None = DEFAULT_REASONING_EFFORT,
        extended_thinking_budget: int | None = DEFAULT_EXTENDED_THINKING_BUDGET,
        reasoning_max_tokens: int | None = None,
        openrouter_provider: str | None = None,
        artifact_root: Path | None = None,
        execution_environment: str = "docker",
        daytona_snapshot: str | None = None,
        daytona_storage_mb: int = 20_480,
    ) -> None:
        self.release = release
        self.output_dir = output_dir.resolve()
        self.tool_root = tool_root.resolve()
        self.lazy_archive_index = (
            self.release.root / "manifests" / "tool-archive-index.json"
        )
        self.verifier_model = verifier_model
        if reasoning_effort is not None and reasoning_effort not in REASONING_EFFORTS:
            raise ReleaseError(f"Unsupported reasoning effort: {reasoning_effort}")
        self.reasoning_effort = reasoning_effort
        self.extended_thinking_budget = extended_thinking_budget
        self.reasoning_max_tokens = reasoning_max_tokens
        self.effective_reasoning_config = effective_reasoning_config(
            reasoning_effort,
            extended_thinking_budget,
            reasoning_max_tokens,
        )
        self.provider_routing = (
            {
                "only": [openrouter_provider],
                "allow_fallbacks": False,
                "require_parameters": True,
            }
            if openrouter_provider
            else None
        )
        self.litellm_extra_body: dict[str, Any] = {}
        if self.provider_routing:
            self.litellm_extra_body["provider"] = self.provider_routing
        if reasoning_max_tokens is not None:
            self.litellm_extra_body["reasoning"] = {
                "enabled": True,
                "max_tokens": reasoning_max_tokens,
            }
        self.artifact_root = artifact_root.resolve() if artifact_root else None
        if execution_environment not in {"docker", "daytona"}:
            raise ReleaseError(f"Unsupported execution environment: {execution_environment}")
        if execution_environment == "daytona" and not daytona_snapshot:
            raise ReleaseError("Daytona materialization requires a prebuilt Daytona snapshot")
        self.execution_environment = execution_environment
        self.daytona_snapshot = daytona_snapshot
        self.daytona_storage_mb = daytona_storage_mb
        self.daytona_identity: dict[str, str] | None = None
        if self.execution_environment == "daytona":
            archive_index = self.release.tool_archive_index()
            self.daytona_identity = {
                "release_revision": self.release.revision,
                "mcp_tree_sha256": str(
                    self.release.tool_profile["source_tree_sha256"]
                ),
                "archive_index_sha256": str(archive_index["index_sha256"]),
                "archive_set_sha256": str(archive_index["archive_set_sha256"]),
                "runtime_sha256": runtime_template_sha256(),
            }
            required_name_tokens = (
                f"r{snapshot_revision_label(self.release.revision)}",
                f"m{self.daytona_identity['mcp_tree_sha256'][:8]}",
                f"i{self.daytona_identity['archive_index_sha256'][:8]}",
                f"a{self.daytona_identity['archive_set_sha256'][:8]}",
                f"t{self.daytona_identity['runtime_sha256'][:8]}",
            )
            if not all(token in str(self.daytona_snapshot) for token in required_name_tokens):
                raise ReleaseError(
                    "Daytona snapshot name does not encode the canonical MCP, "
                    "archive-index, archive-set, and runtime digests"
                )
        if not TEMPLATE_DIR.is_dir():
            raise ReleaseError(f"Missing BVB Harbor template: {TEMPLATE_DIR}")
        if self.execution_environment == "docker" and not self.tool_root.is_dir():
            raise ReleaseError(f"Extracted shared-tool root does not exist: {self.tool_root}")

    def generate(self, tasks: list[BVBTask]) -> list[MaterializedTask]:
        """Generate caller-controlled build output without modifying the release."""
        if (self.output_dir / "bvb-build-manifest.json").exists():
            raise ReleaseError(f"Refusing to overwrite existing build output: {self.output_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        results = [self._generate_task(task) for task in tasks]
        (self.output_dir / "bvb-build-manifest.json").write_text(
            json.dumps(
                {
                    "release_root": str(self.release.root),
                    "verifier_model": self.verifier_model,
                    "effective_reasoning_configuration": self.effective_reasoning_config,
                    "provider_routing": self.provider_routing,
                    "litellm_extra_body": self.litellm_extra_body or None,
                    "tasks": [
                        {
                            "task_id": result.task_id,
                            "task_dir": str(result.task_dir),
                            "trajectory_provenance": result.trajectory_provenance,
                            "workspace_provenance": result.workspace_provenance,
                        }
                        for result in results
                    ],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        return results

    def _generate_task(self, task: BVBTask) -> MaterializedTask:
        task_dir = self.output_dir / task.task_id
        if task_dir.exists():
            raise ReleaseError(f"Refusing to overwrite existing generated task: {task_dir}")
        shutil.copytree(
            TEMPLATE_DIR,
            task_dir,
            ignore=shutil.ignore_patterns("_tests-*", "_environment-*"),
        )
        self._overlay_verifier_template(task_dir)
        self._replace_mcp_tree(task_dir)
        local_runtime_fixes = self._configure_local_runtime(task_dir)
        self._write_instruction_and_config(task_dir, task)
        self._write_environment_input(task_dir, task)
        workspace_provenance, trajectory_provenance, artifact_metadata = self._write_solution(
            task_dir, task
        )
        validate_rubric(json.loads((task_dir / "tests/rubric.json").read_text()), canonical_rows(task))
        guidance_name = "judge-guidance.md" if self.verifier_kind == "gandalf" else "judge-prompt.md"
        guidance_path = task_dir / "tests" / guidance_name
        guidance_path.write_text(guidance_path.read_text() + "\n" + REPLAY_EVIDENCE_GUIDANCE)
        contract = {"identity": input_identity(self.release, task), "rubric": canonical_rows(task),
                    "instructions": self.release.instruction_for(task)}
        (task_dir / "tests/input-contract.json").write_text(json.dumps(contract, indent=2) + "\n")
        shutil.copy2(Path(__file__).with_name("contract.py"), task_dir / "tests/bvb_contract.py")
        generated_manifest = task_dir / "bvb-task-manifest.json"
        generated_manifest.write_text(
            json.dumps(
                {
                    "schema_version": "bvb-v1.0-generated-task/v1",
                    "release_name": self.release.release_id,
                    "dataset_revision": self.release.revision,
                    "cohort": self.release.cohort,
                    "task_id": task.task_id,
                    "instruction_sha256": sha256_file(task_dir / "instruction.md"),
                    "input_tree_sha256": tree_digest(task_dir / "environment" / "input"),
                    "canonical_rubric_sha256": task.manifest["evaluation_rubric"]["sha256"]
                    if task.criteria
                    else None,
                    "canonical_rubric_count": len(task.criteria),
                    "mcp_tree_sha256": tree_digest(task_dir / "environment" / "mcp-server"),
                    "tool_root": str(self.tool_root),
                    "execution_environment": self.execution_environment,
                    "local_runtime_fixes": local_runtime_fixes,
                    "daytona_snapshot": self.daytona_snapshot,
                    "daytona_storage_mb": self.daytona_storage_mb,
                    "daytona_snapshot_identity": self.daytona_identity,
                    "verifier_kind": self.verifier_kind,
                    "verifier_model": self.verifier_model,
                    "effective_reasoning_configuration": self.effective_reasoning_config,
                    "provider_routing": self.provider_routing,
                    "litellm_extra_body": self.litellm_extra_body or None,
                    **self._generated_verifier_metadata(),
                    "selected_rollout": artifact_metadata,
                    "workspace_provenance": workspace_provenance,
                    "trajectory_provenance": trajectory_provenance,
                    "verifier_workdir": self._judge_workdir(),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        return MaterializedTask(
            task_id=task.task_id,
            task_dir=task_dir,
            workspace_provenance=workspace_provenance,
            trajectory_provenance=trajectory_provenance,
            generated_manifest=generated_manifest,
        )

    def _overlay_verifier_template(self, task_dir: Path) -> None:
        """Hook for verifier variants that replace tests or image files."""

    def _generated_verifier_metadata(self) -> dict[str, Any]:
        return {}

    def _configure_local_runtime(self, task_dir: Path) -> dict[str, Any]:
        """Repair local exports without changing the release or measured Daytona base."""
        if self.execution_environment != "docker":
            return {}
        from .runtime_snapshots import LOGO_EXPORT_PERMISSIONS

        fix = dict(LOGO_EXPORT_PERMISSIONS)
        # Reproduce the existing derived-runtime patch exactly, with both hashes
        # checked before writing. Apply it inside the image: the canonical MCP
        # source copied into the build context remains byte-identical to HF.
        original = "            shutil.copy2(str(src), str(dest_file))\n"
        replacement = original + (
            "            # Exported evidence is shared; the original cache stays private.\n"
            "            dest_file.chmod(0o640)\n"
        )
        program = "; ".join([
            "import hashlib; from pathlib import Path",
            f"path = Path({fix['path']!r})",
            "source = path.read_bytes()",
            f"assert hashlib.sha256(source).hexdigest() == {fix['original_sha256']!r}, 'Unexpected MCP logo source'",
            f"patched = source.replace({original.encode()!r}, {replacement.encode()!r})",
            f"assert hashlib.sha256(patched).hexdigest() == {fix['patched_sha256']!r}, 'Unexpected MCP logo patch'",
            "path.write_bytes(patched)",
        ])
        dockerfile = task_dir / "environment" / "Dockerfile"
        with dockerfile.open("a", encoding="utf-8") as stream:
            stream.write(
                "\n# Local runtime correction: group-readable logo exports (recorded in bvb-task-manifest.json).\n"
                "USER root\n"
                f"RUN /opt/mcp-server/.venv/bin/python -c {shlex.quote(program)}\n"
                "USER agent\n"
            )
        return {"logo_export_permissions": fix}

    def _replace_mcp_tree(self, task_dir: Path) -> None:
        destination = task_dir / "environment" / "mcp-server"
        shutil.rmtree(destination, ignore_errors=True)
        shutil.copytree(self.release.mcp_root, destination)
        expected = str(self.release.tool_profile["source_tree_sha256"])
        actual = tree_digest(destination)
        if actual != expected:
            raise ReleaseError("Copied canonical MCP tree does not match tool-profile.json")

    def _write_instruction_and_config(self, task_dir: Path, task: BVBTask) -> None:
        instruction = self.release.instruction_for(task)
        (task_dir / "instruction.md").write_text(instruction, encoding="utf-8")
        rubric = [
            {
                "criterion": criterion.text,
                "weight": criterion.weight,
                "category": criterion.category,
                "evaluation_criterion_index": criterion.evaluation_index,
                "source_criterion_index": criterion.source_index,
                "criterion_text_sha256": criterion.text_sha256,
            }
            for criterion in task.criteria
        ]
        (task_dir / "tests" / "rubric.json").write_text(
            json.dumps(rubric, indent=2) + "\n", encoding="utf-8"
        )
        self._write_verifier_config(task_dir, task, instruction, rubric)
        task_toml = {
            "version": "1.0",
            "metadata": {
                "bvb_release": self.release.release_id,
                "bvb_task_id": task.task_id,
                "bvb_cohort_eligible": task.is_adjudicated,
                "bvb_execution_environment": self.execution_environment,
                "bvb_local_runtime_fixes": ["logo_export_permissions"] if self.execution_environment == "docker" else [],
                "bvb_daytona_snapshot": self.daytona_snapshot or "",
                "verifier_kind": self.verifier_kind,
                "verifier_model": self.verifier_model,
                **self._task_verifier_metadata(),
            },
            "verifier": self._verifier_task_config(),
            "agent": {"timeout_sec": 3000.0, "user": "agent"},
            "environment": {
                "build_timeout_sec": 600.0,
                "cpus": 4,
                "memory_mb": 8192,
                "storage_mb": (
                    self.daytona_storage_mb
                    if self.execution_environment == "daytona"
                    else 10240
                ),
                "gpus": 0,
                # Live judging still needs outbound access to its configured
                # provider. Task-local data retrieval uses the mounted MCP.
                "allow_internet": True,
                "mcp_servers": [_mcp_server_config(self.execution_environment, lazy_tools=self.lazy_archive_index.is_file())],
            },
            "solution": {"env": {}},
        }
        if self.execution_environment == "daytona":
            # Remote agent logs are collected before verification. Export the
            # actual workspace after verification rather than relying on mounts.
            task_toml["artifacts"] = [{"source": "/home/agent/workspace",
                                       "destination": "reviewed-workspace"}]
        (task_dir / "task.toml").write_bytes(tomli_w.dumps(task_toml).encode())
        self._write_compose_config(task_dir)

    def _task_verifier_metadata(self) -> dict[str, Any]:
        return {}

    def _judge_workdir(self) -> str:
        return REPLAY_VIEW

    def _write_verifier_config(
        self,
        task_dir: Path,
        task: BVBTask,
        instruction: str,
        rubric: list[dict[str, Any]],
    ) -> None:
        """Write Gandalf's in-environment verifier configuration."""
        mcp_server = _mcp_server_config(self.execution_environment, lazy_tools=self.lazy_archive_index.is_file())
        grader = {
            "model": self.verifier_model,
            "sandbox_user": "sandbox",
            "instructions": instruction,
            "rubric_path": "/tests/rubric.json",
            "judge_guidance_path": "/tests/judge-guidance.md",
            "output_dir": "/logs/verifier",
            "workdir": self._judge_workdir(),
            "trajectory_path": "/logs/agent/trajectory.json",
            "judge_timeout": 300,
            "judge_retries": 1,
            "batch_timeout": 1740,
            "mode": "batch",
            "batch_splits": 2,
            "max_concurrency": 2,
            "mcp_servers": [mcp_server],
        }
        (task_dir / "tests" / "grader.toml").write_bytes(tomli_w.dumps(grader).encode())
        self._write_judge_configuration(task_dir)

    def _verifier_task_config(self) -> dict[str, Any]:
        return {
            "timeout_sec": 1800.0,
            "user": "verifier",
            "env": {"LLM_API_KEY": f"${{{_api_key_env_var(self.verifier_model)}}}"},
        }

    def _write_compose_config(self, task_dir: Path) -> None:
        compose_path = task_dir / "environment" / "docker-compose.yaml"
        if self.execution_environment == "daytona":
            # Harbor uses direct Daytona sandboxes when compose is absent.
            # Shared tools live in the immutable snapshot rather than a
            # non-portable local bind mount.
            compose_path.unlink(missing_ok=True)
        else:
            compose = {
                "services": {
                    "main": {
                        "volumes": [
                            (
                                f"{self.lazy_archive_index}:"
                                "/opt/mcp-server/data/tool-archive-index.json:ro"
                            ),
                            (
                                f"{self.release.shared_tools_dir}:"
                                "/opt/mcp-server/data/shared-tools:ro"
                            ),
                        ],
                        "environment": {
                            "BVB_TOOL_ARCHIVE_INDEX": (
                                "/opt/mcp-server/data/tool-archive-index.json"
                            ),
                            "BVB_TOOL_ARCHIVE_ROOT": "/opt/mcp-server/data",
                            "BVB_TOOL_CACHE_ROOT": "/var/cache/bvb-tools",
                            "BVB_TOOL_CACHE_MAX_BYTES": "2147483648",
                        },
                    }
                }
            }
            compose_path.write_text(yaml.safe_dump(compose, sort_keys=False), encoding="utf-8")

    def _write_judge_configuration(self, task_dir: Path) -> None:
        sitecustomize = task_dir / "tests" / "sitecustomize.py"
        extra_body_lines = (
            "    extra_body = dict(kwargs.get('litellm_extra_body') or {})\n"
            f"    extra_body.update({self.litellm_extra_body!r})\n"
            "    kwargs['litellm_extra_body'] = extra_body\n"
            if self.litellm_extra_body
            else ""
        )
        sitecustomize.write_text(
            (
                "try:\n"
                "    from openhands.sdk import LLM\n"
                "except ImportError:\n"
                "    LLM = None\n\n"
                "if LLM is not None and not getattr(LLM, '_bvb_configured', False):\n"
                "    _original_init = LLM.__init__\n\n"
                "    def _configured_init(self, *args, **kwargs):\n"
                f"        kwargs['reasoning_effort'] = {self.reasoning_effort!r}\n"
                "        kwargs['extended_thinking_budget'] = "
                f"{self.extended_thinking_budget!r}\n"
                f"{extra_body_lines.replace('    ', '        ')}"
                "        _original_init(self, *args, **kwargs)\n\n"
                "    LLM.__init__ = _configured_init\n"
                "    LLM._bvb_configured = True\n"
            ),
            encoding="utf-8",
        )

    def _write_environment_input(self, task_dir: Path, task: BVBTask) -> None:
        destination = task_dir / "environment" / "input"
        shutil.rmtree(destination, ignore_errors=True)
        source = self.release.root / "task-data" / task.task_id / "Inputs"
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            destination.mkdir(parents=True)
        expected_inputs = task.manifest.get("inputs", [])
        if len(expected_inputs) != sum(1 for path in destination.rglob("*") if path.is_file()):
            raise ReleaseError(f"{task.task_id} input count does not match the task manifest")
        for entry in expected_inputs:
            candidate = destination / entry["path"]
            if not candidate.is_file() or sha256_file(candidate) != entry["sha256"]:
                raise ReleaseError(f"{task.task_id} input hash mismatch: {entry['path']}")

    def _write_solution(self, task_dir: Path, task: BVBTask) -> tuple[str, str, dict[str, Any] | None]:
        solution = task_dir / "solution"
        workspace = solution / "workspace"
        workspace.mkdir(parents=True)
        selected = task.selected_rollout
        if selected is None:
            raise ReleaseError(f"{task.task_id} has no human-reviewed selected rollout")

        candidate_root = self.release.root / selected.candidate_path
        if tree_digest(candidate_root) != selected.candidate_tree_sha256:
            raise ReleaseError(f"{task.task_id} selected candidate tree hash mismatch")
        agent_dir = _find_authentic_agent_dir(task, candidate_root, self.artifact_root)
        declared_workspace_hash, declared_trajectory_hash = _declared_replay_hashes(task)
        if not (
            agent_dir is not None
            and (agent_dir / "workspace").is_dir()
            and (agent_dir / "trajectory.json").is_file()
            and declared_workspace_hash is not None
            and declared_trajectory_hash is not None
            and tree_digest(agent_dir / "workspace") == declared_workspace_hash
            and sha256_file(agent_dir / "trajectory.json") == declared_trajectory_hash
        ):
            raise ReleaseError(f"{task.task_id} requires matching authentic selected workspace and trajectory")
        _copytree(agent_dir / "workspace", workspace)
        trajectory_destination = solution / "trajectory.json"
        shutil.copy2(agent_dir / "trajectory.json", trajectory_destination)
        self._write_solve_script(solution)
        return (
            "authentic_agent_workspace",
            "authentic_agent_trajectory",
            {
                "model": selected.model,
                "trial_name": selected.trial_name,
                "candidate_path": selected.candidate_path,
                "candidate_tree_sha256": selected.candidate_tree_sha256,
                "authentic_agent_dir": str(agent_dir) if agent_dir else None,
                "declared_workspace_tree_sha256": declared_workspace_hash,
                "declared_trajectory_sha256": declared_trajectory_hash,
                "workspace_sha256": tree_digest(workspace),
                "trajectory_sha256": sha256_file(trajectory_destination),
            },
        )

    @staticmethod
    def _write_solve_script(solution: Path) -> None:
        lines = [
            "#!/bin/bash",
            "set -euo pipefail",
            "mkdir -p /home/agent/workspace",
            "cp -a /solution/workspace/. /home/agent/workspace/",
            "mkdir -p /logs/agent",
            "cp /solution/trajectory.json /logs/agent/trajectory.json",
        ]
        solve = solution / "solve.sh"
        solve.write_text("\n".join(lines) + "\n", encoding="utf-8")
        solve.chmod(0o755)


class RewardKitBVBAdapter(BVBAdapter):
    """Generate a separate BVB task shape that runs RewardKit in Harbor."""

    verifier_kind = "rewardkit"

    def __init__(
        self,
        release: BVBRelease,
        *,
        output_dir: Path,
        tool_root: Path,
        verifier_model: str = DEFAULT_REWARDKIT_VERIFIER_MODEL,
        rewardkit_agent: str = DEFAULT_REWARDKIT_AGENT,
        rewardkit_reasoning_effort: str = "high",
        rewardkit_judge_timeout_sec: int = DEFAULT_REWARDKIT_JUDGE_TIMEOUT_SEC,
        harbor_verifier_timeout_sec: float = DEFAULT_REWARDKIT_HARBOR_TIMEOUT_SEC,
        artifact_root: Path | None = None,
        execution_environment: str = "docker",
        daytona_snapshot: str | None = None,
        daytona_storage_mb: int = 20_480,
    ) -> None:
        if rewardkit_agent not in {"claude-code", "codex"}:
            raise ReleaseError(
                "RewardKit 0.1.2 agent judges support claude-code or codex."
            )
        if rewardkit_reasoning_effort not in REASONING_EFFORTS:
            raise ReleaseError("Unsupported RewardKit reasoning effort")
        if rewardkit_agent == "codex" and not verifier_model.startswith("openai/"):
            raise ReleaseError("RewardKit Codex currently requires an OpenAI model")
        if rewardkit_agent == "codex" and execution_environment != "daytona":
            raise ReleaseError("RewardKit Codex requires the verified Daytona runtime snapshot")
        if execution_environment == "daytona" and "-rewardkit-" not in str(daytona_snapshot):
            raise ReleaseError("RewardKit requires its own verified Daytona snapshot")
        if rewardkit_judge_timeout_sec >= int(harbor_verifier_timeout_sec):
            raise ReleaseError(
                "RewardKit's judge timeout must be lower than Harbor's verifier timeout."
            )
        self.rewardkit_agent = rewardkit_agent
        self.rewardkit_reasoning_effort = rewardkit_reasoning_effort
        self.rewardkit_version = DEFAULT_REWARDKIT_VERSION
        self.rewardkit_judge_timeout_sec = rewardkit_judge_timeout_sec
        self.harbor_verifier_timeout_sec = harbor_verifier_timeout_sec
        super().__init__(
            release,
            output_dir=output_dir,
            tool_root=tool_root,
            verifier_model=verifier_model,
            artifact_root=artifact_root,
            execution_environment=execution_environment,
            daytona_snapshot=daytona_snapshot,
            daytona_storage_mb=daytona_storage_mb,
        )
        self.effective_reasoning_config = {
            "agent": rewardkit_agent,
            "isolated": False,
            "reasoning_effort": rewardkit_reasoning_effort,
            "boundary": "private replay copy, inner sandbox UID 10003",
            "source": "harbor-rewardkit agent judge",
        }
        self.provider_routing = None

    def _overlay_verifier_template(self, task_dir: Path) -> None:
        tests_dir = task_dir / "tests"
        shutil.rmtree(tests_dir)
        shutil.copytree(TEMPLATE_DIR / "_tests-rewardkit", tests_dir)
        shutil.copy2(
            TEMPLATE_DIR / "_environment-rewardkit" / "Dockerfile",
            task_dir / "environment" / "Dockerfile",
        )
        test_script = tests_dir / "test.sh"
        test_script.chmod(0o755)

    def _generated_verifier_metadata(self) -> dict[str, Any]:
        return {
            "rewardkit_agent": self.rewardkit_agent,
            "rewardkit_reasoning_effort": self.rewardkit_reasoning_effort,
            "rewardkit_version": self.rewardkit_version,
            "rewardkit_judge_timeout_sec": self.rewardkit_judge_timeout_sec,
            "harbor_verifier_timeout_sec": self.harbor_verifier_timeout_sec,
        }

    def _task_verifier_metadata(self) -> dict[str, Any]:
        return {
            "rewardkit_agent": self.rewardkit_agent,
            "rewardkit_version": self.rewardkit_version,
        }

    def _write_verifier_config(
        self,
        task_dir: Path,
        task: BVBTask,
        instruction: str,
        rubric: list[dict[str, Any]],
    ) -> None:
        del task
        prompt_path = task_dir / "tests/judge-prompt.md"
        prompt_path.write_text("Task prompt:\n" + instruction + "\n\n" + prompt_path.read_text())
        (task_dir / "tests/mcp-config.json").write_text(json.dumps(_mcp_server_config(self.execution_environment, lazy_tools=self.lazy_archive_index.is_file()), indent=2))
        criteria_path = task_dir / "tests" / "rubric" / "criteria.toml"
        criteria_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Generated BVB v1.0 RewardKit rubric.",
            "[judge]",
            f'judge = "{self.rewardkit_agent}"',
            f'model = "{self.verifier_model}"',
            f"timeout = {self.rewardkit_judge_timeout_sec}",
            "isolated = false",
            'atif-trajectory = "/logs/agent/trajectory.json"',
            'prompt_template = "../judge-prompt.md"',
        ]
        seen: dict[str, int] = {}
        for item in rubric:
            base_name = _slugify(str(item["criterion"]))
            seen[base_name] = seen.get(base_name, 0) + 1
            name = base_name if seen[base_name] == 1 else f"{base_name}_{seen[base_name]}"
            description = (
                str(item["criterion"]).replace("\\", "\\\\").replace('"', '\\"')
            )
            lines.extend(
                [
                    "",
                    "[[criterion]]",
                    f'name = "{name}"',
                    f'description = {json.dumps(str(item["criterion"]))}',
                    'type = "binary"',
                    f'weight = {float(item.get("weight", 1))}',
                ]
            )
        parsed = tomllib.loads("\n".join(lines))
        if self.rewardkit_agent == "codex":
            parsed["judge"]["model"] = self.verifier_model.removeprefix("openai/")
        for entry, item in zip(parsed["criterion"], rubric):
            entry["description"] = item["criterion"]
        criteria_path.write_text(tomli_w.dumps(parsed))

    def _verifier_task_config(self) -> dict[str, Any]:
        api_key_env = _api_key_env_var(self.verifier_model)
        return {
            "timeout_sec": self.harbor_verifier_timeout_sec,
            "user": "verifier",
            "env": {
                api_key_env: f"${{{api_key_env}}}",
                "BVB_REWARDKIT_AGENT": self.rewardkit_agent,
                "BVB_REWARDKIT_REASONING_EFFORT": self.rewardkit_reasoning_effort,
                "OTEL_EXPORTER_OTLP_ENDPOINT": "${OTEL_EXPORTER_OTLP_ENDPOINT:-}",
                "OTEL_EXPORTER_OTLP_HEADERS": "${OTEL_EXPORTER_OTLP_HEADERS:-}",
                "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL": (
                    "${OTEL_EXPORTER_OTLP_TRACES_PROTOCOL:-}"
                ),
            },
        }


def _slugify(text: str, max_len: int = 50) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return (slug[:max_len] or "criterion").rstrip("_")

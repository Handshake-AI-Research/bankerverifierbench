"""Gandalf criterion-batching conditions (B-b4 ... B-b64, B-split2).

Gandalf 1.0.0 only splits a rubric into N near-equal batches (`batch_splits`).
The batching study also needs a fixed number of criteria per batch, so each
task carries a copy of the pinned Gandalf 1.0.0 package with an exact-size
`batch_size` mode, plus logging hooks that record each judge session's actual
input. The judge prompt, templates and verdict parsing are unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import tomllib
import urllib.request
import zipfile

import tomli_w

RUNTIME = Path(__file__).with_name("batching_runtime")
NATIVE_FILES = RUNTIME / "native-files.json"
GANDALF_WHEEL = dict(
    url="https://files.pythonhosted.org/packages/e7/d3/3af8f410ccea8525edd814d0de8609f34f544bff8c61e62a623a7d839d06/"
        "gandalf_the_grader-1.0.0-py3-none-any.whl",
    sha256="486cff4222cca70c2637b3c62e306edebeedbdd467108ec27c28e77003f620c7",
)
MODEL = "gemini/gemini-3-flash-preview"
REASONING_EFFORT = "high"
VERIFIER_TIMEOUT_SEC = 21600.0
EXPECTED_GRADER = dict(max_concurrency=2, batch_timeout=1740, judge_timeout=300, judge_retries=1, batch_splits=2)


@dataclass(frozen=True)
class BatchingCondition:
    name: str
    batch_size: int | None = None
    batch_splits: int | None = None

    def __post_init__(self):
        if (self.batch_size is None) == (self.batch_splits is None):
            raise ValueError("Set exactly one of batch_size or batch_splits")
        if self.batch_splits is not None and self.batch_splits != 2:
            raise ValueError("Only the measured batch_splits = 2 control is supported")
        if self.batch_size is not None and self.batch_size < 1:
            raise ValueError("batch_size must be positive")

    def partition(self, criteria: list) -> list[list]:
        size = self.batch_size or math.ceil(len(criteria) / self.batch_splits)
        return [criteria[i:i + size] for i in range(0, len(criteria), size)]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def native_source(destination: Path) -> Path:
    """Extract the pinned Gandalf 1.0.0 package files from its PyPI wheel."""
    with urllib.request.urlopen(GANDALF_WHEEL["url"], timeout=120) as response:
        wheel = response.read()
    if _sha256(wheel) != GANDALF_WHEEL["sha256"]:
        raise ValueError("Gandalf 1.0.0 wheel hash differs")
    expected = json.loads(NATIVE_FILES.read_text())
    with zipfile.ZipFile(io.BytesIO(wheel)) as archive:
        for name, digest in expected.items():
            data = archive.read("gandalf/" + name)
            if _sha256(data) != digest:
                raise ValueError(f"Gandalf 1.0.0 source differs: {name}")
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    return destination


def build_overlay(source: Path, destination: Path) -> Path:
    """Add exact-size batching and input observation to pinned Gandalf 1.0.0."""
    for name, digest in json.loads(NATIVE_FILES.read_text()).items():
        if _sha256((source / name).read_bytes()) != digest:
            raise ValueError(f"Gandalf 1.0.0 source differs: {name}")
    shutil.copytree(source, destination)

    def replace(name, old, new):
        path = destination / name
        text = path.read_text()
        if text.count(old) != 1:
            raise ValueError(f"Integration context differs in {name}: {old!r}")
        path.write_text(text.replace(old, new))

    replace("models.py", "    batch_splits: int | None = Field(default=None, ge=2)",
            "    batch_splits: int | None = Field(default=None, ge=2)\n"
            "    batch_size: int | None = Field(default=None, ge=1)")
    replace("models.py", "        return self\n",
            "        if self.batch_size is not None and (self.mode != \"batch\" or self.batch_splits is not None):\n"
            "            raise ValueError(\"batch_size requires batch mode and cannot be combined with batch_splits\")\n"
            "        return self\n")
    replace("orchestrator.py", "    chunk_size = math.ceil(n / splits)",
            "    chunk_size = config.batch_size or math.ceil(n / splits)")
    replace("orchestrator.py", "        run = run_batch_concurrent if splits is not None else run_batch",
            "        run = run_batch_concurrent if splits is not None or config.batch_size is not None else run_batch")
    replace("orchestrator.py", "    retry_run = run_batch if config.mode == \"batch\" else run_individual",
            "    retry_run = (run_batch_concurrent if config.batch_size is not None else run_batch) "
            "if config.mode == \"batch\" else run_individual")
    replace("orchestrator.py", "    if config.mode == \"batch\" and config.batch_splits is not None:\n        mode_display",
            "    if config.batch_size is not None:\n"
            "        mode_display = f\"batch (criteria_per_batch={config.batch_size}, max_concurrency={concurrency})\"\n"
            "    elif config.mode == \"batch\" and config.batch_splits is not None:\n        mode_display")
    replace("orchestrator.py", "        os.chmod(input_path, 0o644)",
            "        os.chmod(input_path, 0o644)\n"
            "        from bvb_batch_observe import observe_clone\n"
            "        observe_clone(cloned_input, trace_path)")

    expected = json.loads((RUNTIME / "overlay-files.json").read_text())
    actual = {str(p.relative_to(destination)): _sha256(p.read_bytes()) for p in destination.rglob("*") if p.is_file()}
    if actual != expected:
        raise ValueError("Batching overlay differs from the measured overlay")
    return destination


def configure_task(task_dir: Path, condition: BatchingCondition, overlay: Path) -> None:
    """Turn a standard Gandalf Daytona task into a batching-condition task."""
    tests = task_dir / "tests"
    grader_path = tests / "grader.toml"
    grader = tomllib.loads(grader_path.read_text())
    if {key: grader.get(key) for key in EXPECTED_GRADER} != EXPECTED_GRADER:
        raise ValueError(f"{task_dir.name}: unexpected native Gandalf batching defaults")
    if condition.batch_size is not None:
        grader.pop("batch_splits")
        grader["batch_size"] = condition.batch_size
    grader_path.write_bytes(tomli_w.dumps(grader).encode())
    task_path = task_dir / "task.toml"
    task = tomllib.loads(task_path.read_text())
    task["verifier"]["timeout_sec"] = VERIFIER_TIMEOUT_SEC
    task_path.write_bytes(tomli_w.dumps(task).encode())
    shutil.copytree(overlay, tests / "gandalf")
    shutil.copy2(RUNTIME / "bvb_batch_observe.py", tests / "bvb_batch_observe.py")
    sitecustomize = tests / "sitecustomize.py"
    sitecustomize.write_text(sitecustomize.read_text() + (RUNTIME / "session-observation.py.txt").read_text())

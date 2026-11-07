"""Immutable BankerVerifierBench release loading and validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .config import (
    COHORT,
    DATASET_MANIFEST_SHA256,
    DATASET_REPOSITORY,
    DATASET_REVISION,
    EXPECTED_TOOL_ROOTS,
    MCP_PROFILE_PATH,
    RELEASE_NAME,
    RELEASE_SCHEMA,
    SHARED_TOOLS_DIR,
)
from .schema import BVBTask, Criterion, SelectedRollout


class ReleaseError(RuntimeError):
    """Raised when a downloaded release violates the BVB data contract."""


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_digest(root: Path) -> str:
    """Hash a tree using release-compatible relative-path and file-hash records."""
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(sha256_file(path).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def validate_rollout_evidence(root: Path) -> None:
    """Reject historical grader outputs and other non-evidence rollout files."""
    files = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if ("agent/trajectory.json" not in files
            or not any(p.startswith("agent/workspace/") for p in files)
            or any(p != "agent/trajectory.json" and not p.startswith("agent/workspace/") for p in files)):
        raise ReleaseError(f"Rollout must contain only workspace and trajectory evidence: {root}")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReleaseError(f"Invalid JSON at {path}: {error}") from error
    if not isinstance(value, dict):
        raise ReleaseError(f"Expected a JSON object at {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, json.JSONDecodeError) as error:
        raise ReleaseError(f"Invalid JSONL at {path}: {error}") from error
    if not all(isinstance(row, dict) for row in rows):
        raise ReleaseError(f"Expected JSON object rows at {path}")
    return rows


def _task_key(task_record: dict[str, Any]) -> str:
    task_id = str(task_record.get("task_id", "")).strip()
    if not task_id:
        raise ReleaseError("tasks.jsonl contains a task without task_id")
    return task_id if task_id.startswith("btb-") else f"btb-{task_id.split('-', maxsplit=1)[0]}"


@dataclass(frozen=True)
class ValidationResult:
    """A summary suitable for CLI output and preflight manifests."""

    task_count: int
    adjudicated_task_count: int
    eligible_criteria_count: int
    missing_files: tuple[str, ...]


class BVBRelease:
    """Read-only view over one downloaded BankerVerifierBench release."""

    def __init__(
        self,
        root: Path,
        release: dict[str, Any],
        tasks: dict[str, BVBTask],
        tool_profile: dict[str, Any],
        coverage: dict[str, Any],
    ) -> None:
        self.root = root
        self.release = release
        self.tasks = tasks
        self.tool_profile = tool_profile
        self.coverage = coverage

    @classmethod
    def open(cls, root: str | Path, *, validate: bool = True) -> "BVBRelease":
        """Open a release after validating its structural data contract."""
        root = Path(root).resolve()
        release = _read_json(root / "release.json")
        if release.get("release_name") != RELEASE_NAME or release.get("schema_version") != RELEASE_SCHEMA:
            raise ReleaseError(f"{root} is not a BankerVerifierBench release")
        task_records = _read_jsonl(root / "tasks.jsonl")
        manifests = _read_jsonl(root / "manifests" / "task-manifest.jsonl")
        coverage = _read_json(root / "manifests" / "human-label-coverage.json")
        tool_profile = _read_json(root / "manifests" / "tool-profile.json")

        records = {_task_key(record): record for record in task_records}
        manifest_by_id = {str(manifest.get("task_id")): manifest for manifest in manifests}
        expected_tasks = int(release["counts"]["tasks_total"])
        if (len(records) != len(task_records) or len(manifest_by_id) != len(manifests)
                or len(records) != expected_tasks or set(records) != set(manifest_by_id)):
            raise ReleaseError("Task records and manifests must contain the same unique release task IDs")

        tasks: dict[str, BVBTask] = {}
        for task_id, record in records.items():
            manifest = manifest_by_id[task_id]
            rubric_data = manifest.get("evaluation_rubric", {})
            rubric_path = rubric_data.get("path")
            criteria: tuple[Criterion, ...] = ()
            labels_path: str | None = None
            if rubric_data.get("status") == "human_adjudication_canonical":
                if not isinstance(rubric_path, str):
                    raise ReleaseError(f"{task_id} has canonical rubric status without a rubric path")
                if sha256_file(root / rubric_path) != rubric_data.get("sha256"):
                    raise ReleaseError(f"{task_id} canonical rubric file hash mismatch")
                rubric = _read_json(root / rubric_path)
                criteria = tuple(
                    Criterion(
                        evaluation_index=int(item["evaluation_criterion_index"]),
                        source_index=int(item["source_criterion_index"]),
                        text=str(item["criterion_text"]),
                        text_sha256=str(item["criterion_text_sha256"]),
                        category=str(item.get("category", "")),
                        weight=item.get("weight", 1),
                    )
                    for item in rubric.get("criteria", [])
                )
                expected_count = int(rubric_data.get("criterion_count", -1))
                if len(criteria) != expected_count:
                    raise ReleaseError(f"{task_id} canonical rubric count does not match its manifest")
                if [criterion.evaluation_index for criterion in criteria] != list(range(len(criteria))):
                    raise ReleaseError(f"{task_id} canonical rubric evaluation indices are not contiguous")
                if len({c.source_index for c in criteria}) != len(criteria):
                    raise ReleaseError(f"{task_id} has duplicate canonical source indices")
                for c in criteria:
                    if hashlib.sha256(c.text.encode()).hexdigest() != c.text_sha256:
                        raise ReleaseError(f"{task_id} canonical criterion text hash mismatch")
                candidate_labels = f"human-adjudication/canonical/{task_id}.json"
                if not (root / candidate_labels).is_file():
                    raise ReleaseError(f"{task_id} canonical rubric exists but canonical labels are absent")
                labels_path = candidate_labels

            target = manifest.get("label_target_rollout")
            selected = None
            if target is not None:
                selected = SelectedRollout(
                    model=str(target["model"]),
                    trial_name=str(target["trial_name"]),
                    candidate_path=str(target["candidate_path"]),
                    candidate_tree_sha256=str(target["candidate_tree_sha256"]),
                )
                candidate_root = root / selected.candidate_path
                if not candidate_root.is_dir():
                    raise ReleaseError(f"{task_id} selected rollout directory is missing: {selected.candidate_path}")

            tasks[task_id] = BVBTask(
                task_id=task_id,
                task_record=record,
                manifest=manifest,
                criteria=criteria,
                selected_rollout=selected,
                labels_path=labels_path,
            )

        instance = cls(root, release, tasks, tool_profile, coverage)
        if validate:
            instance.validate_structure()
        return instance

    @property
    def cohort(self) -> str:
        return COHORT

    @property
    def repository(self) -> str:
        return str(self.release["repository"])

    @property
    def revision(self) -> str:
        return DATASET_REVISION

    @property
    def release_id(self) -> str:
        return str(self.release["release_name"])

    @property
    def mcp_root(self) -> Path:
        return self.root / MCP_PROFILE_PATH

    @property
    def shared_tools_dir(self) -> Path:
        return self.root / SHARED_TOOLS_DIR

    def shared_tools(self) -> dict[str, Any]:
        """BankerToolBench source and hashes of the tool archives this release uses."""
        return _read_json(self.root / "manifests" / "shared-tools.json")

    @property
    def tool_archive_index_path(self) -> Path:
        return self.root / "manifests" / "tool-archive-index.json"

    def tool_archive_index(self, *, require_archive_files: bool = False) -> dict[str, Any]:
        """Load the optional lazy-tool index, validating it against this release."""
        from .tool_archive_index import (
            load_tool_archive_index,
            validate_tool_archive_index,
        )

        if not self.tool_archive_index_path.is_file():
            raise ReleaseError(
                "Release has no manifests/tool-archive-index.json; "
                "use a BVB revision built for compressed Daytona snapshots"
            )
        index = load_tool_archive_index(
            self.tool_archive_index_path,
            tool_profile=self.tool_profile,
        )
        if require_archive_files:
            validate_tool_archive_index(
                index,
                tool_profile=self.tool_profile,
                require_archive_files=self.root,
            )
        return index

    def instruction_for(self, task: BVBTask) -> str:
        return (self.root / "task-source" / task.task_id / "instruction.md").read_text(encoding="utf-8")

    def canonical_labels(self, task: BVBTask) -> dict[str, Any]:
        if task.labels_path is None:
            raise ReleaseError(f"{task.task_id} has no canonical human labels")
        return _read_json(self.root / task.labels_path)

    def task_ids(self, cohort: str = COHORT) -> list[str]:
        if cohort != self.cohort:
            raise ReleaseError(f"Unknown BVB cohort {cohort!r}")
        return sorted(task.task_id for task in self.tasks.values() if task.is_adjudicated)

    def select_tasks(self, cohort: str = COHORT, task_ids: Iterable[str] | None = None) -> list[BVBTask]:
        selected = self.task_ids(cohort)
        if task_ids is not None:
            requested = list(task_ids)
            missing = sorted(set(requested) - set(selected))
            if missing:
                raise ReleaseError(f"Tasks are not available in {cohort}: {', '.join(missing)}")
            selected = requested
        return [self.tasks[task_id] for task_id in selected]

    def validate_rollout_selection(self) -> None:
        """Require exactly the human-reviewed rollout per task, including on disk."""
        selected = {tid: task.selected_rollout.candidate_path for tid, task in self.tasks.items()
                    if task.selected_rollout is not None}
        candidates = _read_jsonl(self.root / "manifests/rollout-candidates.jsonl")
        if len(selected) != len(self.tasks) or len(candidates) != len(selected):
            raise ReleaseError("Expected exactly one released rollout per task")
        if {c["path"] for c in candidates} != set(selected.values()):
            raise ReleaseError("Released rollouts differ from the human-reviewed selection")
        for candidate in candidates:
            if (candidate["path"] != selected.get(candidate["task_id"])
                    or candidate["availability"] != "available"
                    or candidate["model"] != self.tasks[candidate["task_id"]].selected_rollout.model):
                raise ReleaseError("Non-selected or unavailable rollout released")
        for tid, task in self.tasks.items():
            copies = task.manifest["rollout_candidates"]
            if (len(copies) != 1 or copies[0]["path"] != selected[tid]
                    or copies[0]["model"] != task.selected_rollout.model
                    or copies[0]["availability"] != "available"):
                raise ReleaseError("Task manifest includes a non-selected rollout")
        directories = {p.relative_to(self.root).as_posix()
                       for p in (self.root / "rollout-candidates").glob("*/*") if p.is_dir()}
        if directories != set(selected.values()):
            raise ReleaseError("Rollout directories differ from the human-reviewed selection")
        for relative in selected.values():
            validate_rollout_evidence(self.root / relative)
        counts = self.release["counts"]
        if not counts["rollout_candidate_slots"] == counts["available_rollout_candidates"] == len(selected):
            raise ReleaseError("Release must contain exactly one available rollout per task")

    def validate_structure(self) -> ValidationResult:
        """Validate release metadata required before task materialization."""
        if self.repository != DATASET_REPOSITORY:
            raise ReleaseError(f"Release does not name the expected dataset {DATASET_REPOSITORY}")
        if self.tool_profile.get("canonical_profile_path") != MCP_PROFILE_PATH.as_posix():
            raise ReleaseError("Tool profile names an unexpected MCP profile path")
        if not self.mcp_root.is_dir():
            raise ReleaseError(f"Canonical MCP profile is missing: {self.mcp_root}")
        expected_roots = set(self.tool_profile.get("expected_extraction_roots", []))
        if expected_roots != {f"{root}/" for root in EXPECTED_TOOL_ROOTS}:
            raise ReleaseError("Tool profile has unexpected extraction roots")
        source_digest = str(self.tool_profile.get("source_tree_sha256", ""))
        if source_digest != tree_digest(self.mcp_root):
            raise ReleaseError("Canonical MCP profile digest does not match tool-profile.json")

        coverage_ids = {str(item.get("task_id")) for item in self.coverage.get("tasks", [])}
        if coverage_ids != set(self.tasks) or len(coverage_ids) != len(self.tasks):
            raise ReleaseError("Human-label coverage manifest does not cover exactly the release tasks")
        adjudicated = self.task_ids(self.cohort)
        counts = self.release["counts"]
        if len(adjudicated) != int(counts["tasks_total"]):
            raise ReleaseError(f"Expected {counts['tasks_total']} adjudicated tasks, found {len(adjudicated)}")
        eligible = sum(len(self.tasks[task_id].criteria) for task_id in adjudicated)
        if eligible != int(counts["scoring_eligible_criteria"]):
            raise ReleaseError(
                f"Expected {counts['scoring_eligible_criteria']} scoring-eligible criteria, found {eligible}"
            )
        self.validate_rollout_selection()
        self.assert_pinned_revision()
        cohort = _read_json(self.root / "manifests/cohort.json")
        if cohort.get("cohort") != self.cohort or set(cohort.get("task_ids", [])) != set(adjudicated):
            raise ReleaseError("Cohort manifest disagrees with canonical task roster")
        return ValidationResult(
            task_count=len(self.tasks),
            adjudicated_task_count=len(adjudicated),
            eligible_criteria_count=eligible,
            missing_files=tuple(self._missing_declared_files()),
        )

    def _missing_declared_files(self) -> list[str]:
        manifest = self.root / "manifests" / "files.sha256"
        if not manifest.is_file():
            raise ReleaseError("Missing manifests/files.sha256")
        missing: list[str] = []
        for line in manifest.read_text(encoding="utf-8").splitlines():
            try:
                _, relative = line.split("  ", maxsplit=1)
            except ValueError as error:
                raise ReleaseError(f"Malformed files.sha256 line: {line}") from error
            if not (self.root / relative).is_file():
                missing.append(relative)
        missing.extend(
            entry["path"] for entry in self.shared_tools()["files"]
            if not (self.root / entry["path"]).is_file()
        )
        return missing

    def validate_hashes(self, *, require_all_files: bool) -> ValidationResult:
        """Verify every downloaded file; optionally require the shared-tool archives too."""
        result = self.validate_structure()
        missing = list(result.missing_files)
        missing_non_tools = [path for path in missing if not path.startswith(f"{SHARED_TOOLS_DIR}/")]
        if missing_non_tools or (require_all_files and missing):
            details = missing_non_tools or missing
            raise ReleaseError(f"Release payload is incomplete: {', '.join(details[:5])}")

        hash_manifest = self.root / "manifests" / "files.sha256"
        for line in hash_manifest.read_text(encoding="utf-8").splitlines():
            digest, relative = line.split("  ", maxsplit=1)
            if sha256_file(self.root / relative) != digest:
                raise ReleaseError(f"Release hash mismatch: {relative}")
        for entry in self.shared_tools()["files"]:
            path = self.root / entry["path"]
            if path.is_file() and (path.stat().st_size != entry["size_bytes"] or sha256_file(path) != entry["sha256"]):
                raise ReleaseError(f"Shared-tool archive differs from BankerToolBench: {entry['path']}")
        return result

    def assert_pinned_revision(self) -> None:
        """Require the pinned checksum manifest and the files that define every judge input."""
        actual = sha256_file(self.root / "manifests/files.sha256")
        if actual != DATASET_MANIFEST_SHA256:
            raise ReleaseError(
                f"Release checksum manifest (sha256 {actual}) differs from the content pinned for "
                f"{DATASET_REPOSITORY}@{DATASET_REVISION}. Delete {self.root} and run `make fetch` again."
            )
        entries = dict(line.split("  ", 1)[::-1] for line in
                       (self.root / "manifests/files.sha256").read_text().splitlines())
        required = ["release.json", "tasks.jsonl", "manifests/task-manifest.jsonl",
                    "manifests/cohort.json", "manifests/human-label-coverage.json",
                    "manifests/tool-profile.json", "manifests/shared-tools.json",
                    "manifests/tool-archive-index.json", "manifests/rollout-candidates.jsonl"]
        for task in self.tasks.values():
            required.extend([task.manifest["evaluation_rubric"]["path"],
                             task.labels_path,
                             f"task-source/{task.task_id}/instruction.md"])
        for relative in required:
            if relative not in entries or sha256_file(self.root / relative) != entries[relative]:
                raise ReleaseError(f"Pinned release input hash mismatch: {relative}")
        shared = {entry["path"]: entry["sha256"] for entry in self.shared_tools()["files"]}
        profile = {entry["path"]: entry["sha256"] for entry in self.tool_profile["shared_tool_archives"]}
        if shared != profile:
            raise ReleaseError("Shared-tool manifest disagrees with the tool profile")

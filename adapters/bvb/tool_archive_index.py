"""Build and validate the versioned BVB shared-tool archive index."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import re
import runpy
import shutil
import subprocess
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Iterator

from .config import EXPECTED_TOOL_ROOTS
from .release import BVBRelease, ReleaseError, sha256_file

INDEX_SCHEMA_VERSION = "bvb-tool-archive-index/v1"
INDEX_PATH = Path("manifests/tool-archive-index.json")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ToolArchiveIndexError(ReleaseError):
    """Raised when a tool archive index is unsafe, inconsistent, or corrupt."""


def stable_json_bytes(value: Any) -> bytes:
    """Serialize JSON identically for index generation and verification."""
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode()


def index_digest(index: dict[str, Any]) -> str:
    """Hash the complete index except for its self-referential digest field."""
    payload = dict(index)
    payload.pop("index_sha256", None)
    return hashlib.sha256(stable_json_bytes(payload)).hexdigest()


def archive_set_digest(archives: list[dict[str, Any]]) -> str:
    """Hash the ordered archive path/digest records."""
    digest = hashlib.sha256()
    for archive in sorted(archives, key=lambda item: str(item["path"])):
        digest.update(str(archive["path"]).encode())
        digest.update(b"\0")
        digest.update(str(archive["sha256"]).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _safe_relative_path(raw: str, *, label: str) -> PurePosixPath:
    path = PurePosixPath(raw)
    if not raw or path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ToolArchiveIndexError(f"Unsafe {label} path: {raw!r}")
    return path


def _archive_format(path: Path) -> str:
    if path.name.endswith(".tar.gz"):
        return "tar.gz"
    if path.name.endswith(".tar.zst"):
        return "tar.zst"
    raise ToolArchiveIndexError(f"Unsupported shared-tool archive format: {path.name}")


@contextlib.contextmanager
def _tar_stream(path: Path) -> Iterator[tarfile.TarFile]:
    process: subprocess.Popen[bytes] | None = None
    source: BinaryIO | None = None
    try:
        if path.name.endswith(".tar.zst"):
            zstd = shutil.which("zstd") or shutil.which("unzstd")
            if zstd is None:
                raise ToolArchiveIndexError(
                    f"Cannot scan {path.name}: install the zstd command-line tool"
                )
            process = subprocess.Popen(
                [zstd, "-dc", "--", str(path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if process.stdout is None:
                raise ToolArchiveIndexError(f"Unable to read decompressor output for {path.name}")
            source = process.stdout
            tar = tarfile.open(fileobj=source, mode="r|")
        else:
            tar = tarfile.open(path, mode="r|gz")
        with tar:
            yield tar
        if source is not None:
            source.close()
        if process is not None:
            stderr = process.stderr.read().decode(errors="replace") if process.stderr else ""
            if process.wait() != 0:
                raise ToolArchiveIndexError(
                    f"zstd decompression failed for {path.name}: {stderr.strip()}"
                )
    except (OSError, tarfile.TarError) as error:
        raise ToolArchiveIndexError(f"Unable to scan {path.name}: {error}") from error
    finally:
        if source is not None and not source.closed:
            source.close()
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()


def _member_digest(handle: BinaryIO) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def _logical_root(logical_path: PurePosixPath) -> str:
    value = logical_path.as_posix()
    matches = [
        root for root in EXPECTED_TOOL_ROOTS if value == root or value.startswith(f"{root}/")
    ]
    if len(matches) != 1:
        raise ToolArchiveIndexError(
            f"Archive member is outside the declared logical roots: {value}"
        )
    return matches[0]


def _scan_archive(
    archive_path: Path,
    archive_relative: str,
    members: dict[str, dict[str, Any]],
    non_tool_members: dict[str, dict[str, Any]],
    declared_non_tool_members: set[str],
) -> tuple[str, int, int]:
    roots: set[str] = set()
    tool_count = 0
    non_tool_count = 0
    with _tar_stream(archive_path) as tar:
        for member in tar:
            member_path = _safe_relative_path(member.name, label="archive member")
            if member.isdir():
                continue
            if member.issym() or member.islnk() or member.isdev() or not member.isfile():
                raise ToolArchiveIndexError(
                    f"Unsupported member type in {archive_relative}: {member.name}"
                )
            logical_path = member_path.as_posix()
            is_non_tool = logical_path in declared_non_tool_members
            if not is_non_tool:
                root = _logical_root(member_path)
                roots.add(root)
            destination = non_tool_members if is_non_tool else members
            if logical_path in destination:
                raise ToolArchiveIndexError(
                    f"Duplicate logical tool path across archives: {logical_path}"
                )
            source = tar.extractfile(member)
            if source is None:
                raise ToolArchiveIndexError(
                    f"Unable to read {member.name} from {archive_relative}"
                )
            with source:
                digest = _member_digest(source)
            destination[logical_path] = {
                "archive_path": archive_relative,
                "member_path": member.name,
                "size_bytes": member.size,
                "sha256": digest,
            }
            if is_non_tool:
                destination[logical_path]["reason"] = "declared_archive_metadata"
                non_tool_count += 1
            else:
                tool_count += 1
    if len(roots) != 1:
        raise ToolArchiveIndexError(
            f"{archive_relative} must contain exactly one logical root; found {sorted(roots)}"
        )
    return next(iter(roots)), tool_count, non_tool_count


def _load_company_names(mcp_root: Path) -> dict[str, str]:
    registry = mcp_root / "tools" / "company_brand_assets_registry.py"
    if not registry.is_file():
        raise ToolArchiveIndexError(f"Missing company brand registry: {registry}")
    values = runpy.run_path(str(registry))
    names = values.get("TICKER_TO_COMPANY")
    if not isinstance(names, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in names.items()
    ):
        raise ToolArchiveIndexError(f"Malformed company brand registry: {registry}")
    return names


def _api_indexes(
    members: dict[str, dict[str, Any]],
    mcp_root: Path,
) -> dict[str, Any]:
    market_files: dict[str, list[str]] = {}
    logo_paths: list[str] = []
    submissions: dict[str, str] = {}
    company_facts: dict[str, str] = {}
    frames: dict[str, str] = {}
    filings: dict[str, dict[str, str]] = {}
    cik_lookup: str | None = None

    for logical_path in sorted(members):
        parts = PurePosixPath(logical_path).parts
        if parts[0] == "market_data_platform" and len(parts) == 3:
            market_files.setdefault(parts[1], []).append(parts[2])
        elif parts[0] == "company_brand_assets" and len(parts) == 2:
            if PurePosixPath(parts[1]).suffix.lower() == ".png":
                logo_paths.append(logical_path)
        elif parts[:2] == ("sec_edgar", "data"):
            sec_parts = parts[2:]
            if sec_parts == ("cik-lookup.json",):
                cik_lookup = logical_path
            elif len(sec_parts) == 2 and sec_parts[0] == "submissions":
                submissions[PurePosixPath(sec_parts[1]).stem] = logical_path
            elif len(sec_parts) == 2 and sec_parts[0] == "company-facts":
                company_facts[PurePosixPath(sec_parts[1]).stem] = logical_path
            elif len(sec_parts) == 2 and sec_parts[0] == "frames":
                frames[PurePosixPath(sec_parts[1]).stem] = logical_path
            elif len(sec_parts) == 3 and sec_parts[0] == "filings":
                filings.setdefault(sec_parts[1], {})[
                    PurePosixPath(sec_parts[2]).stem
                ] = logical_path

    company_names = _load_company_names(mcp_root)
    logos: list[dict[str, Any]] = []
    for logical_path in logo_paths:
        filename = PurePosixPath(logical_path).name
        stem = PurePosixPath(filename).stem
        is_ticker = not any(character.islower() for character in stem)
        entry: dict[str, Any] = {
            "logical_path": logical_path,
            "filename": filename,
            "stem": stem,
            "company": company_names.get(stem, stem)
            if is_ticker
            else stem.replace("_", " "),
        }
        if is_ticker:
            entry["ticker"] = stem
        logos.append(entry)

    return {
        "market_data_platform": {
            "tickers": sorted(
                ticker for ticker in market_files if not ticker.startswith((".", "_"))
            ),
            "files_by_ticker": {
                ticker: sorted(files) for ticker, files in sorted(market_files.items())
            },
        },
        "company_brand_assets": {"logos": logos},
        "sec_edgar": {
            "cik_lookup": cik_lookup,
            "submissions": submissions,
            "company_facts": company_facts,
            "frames": frames,
            "filings": filings,
        },
    }


def build_tool_archive_index(
    release: BVBRelease,
    *,
    mcp_root: Path | None = None,
) -> dict[str, Any]:
    """Scan the exact declared release archives and return their canonical index."""
    release.validate_hashes(require_all_files=True)
    mcp_root = (mcp_root or release.mcp_root).resolve()
    archives: list[dict[str, Any]] = []
    members: dict[str, dict[str, Any]] = {}
    non_tool_members: dict[str, dict[str, Any]] = {}
    declared_non_tool_members = set(
        release.tool_profile.get("non_tool_archive_members", [])
    )
    declared_archives = list(release.tool_profile.get("shared_tool_archives", []))
    if len(declared_archives) != 19:
        raise ToolArchiveIndexError(
            f"Expected 19 declared shared-tool archives, found {len(declared_archives)}"
        )

    for declared in declared_archives:
        relative = str(declared["path"])
        _safe_relative_path(relative, label="archive")
        archive_path = release.root / relative
        if not archive_path.is_file():
            raise ToolArchiveIndexError(f"Missing shared-tool archive: {relative}")
        actual_digest = sha256_file(archive_path)
        if actual_digest != declared.get("sha256"):
            raise ToolArchiveIndexError(f"Shared-tool archive hash mismatch: {relative}")
        logical_root, tool_count, non_tool_count = _scan_archive(
            archive_path,
            relative,
            members,
            non_tool_members,
            declared_non_tool_members,
        )
        archives.append(
            {
                "path": relative,
                "sha256": actual_digest,
                "size_bytes": archive_path.stat().st_size,
                "format": _archive_format(archive_path),
                "logical_root": f"{logical_root}/",
                "member_count": tool_count + non_tool_count,
                "tool_member_count": tool_count,
                "non_tool_member_count": non_tool_count,
            }
        )
    if set(non_tool_members) != declared_non_tool_members:
        missing = sorted(declared_non_tool_members - set(non_tool_members))
        raise ToolArchiveIndexError(
            f"Declared non-tool archive members were not found: {missing}"
        )

    index: dict[str, Any] = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "tool_profile_source_tree_sha256": str(
            release.tool_profile["source_tree_sha256"]
        ),
        "archive_set_sha256": archive_set_digest(archives),
        "archives": archives,
        "members": members,
        "non_tool_members": non_tool_members,
        "api_indexes": _api_indexes(members, mcp_root),
    }
    index["index_sha256"] = index_digest(index)
    validate_tool_archive_index(
        index,
        tool_profile=release.tool_profile,
        require_archive_files=release.root,
    )
    return index


def validate_tool_archive_index(
    index: dict[str, Any],
    *,
    tool_profile: dict[str, Any] | None = None,
    require_archive_files: Path | None = None,
) -> None:
    """Fail closed when an archive index violates the v1 contract."""
    if index.get("schema_version") != INDEX_SCHEMA_VERSION:
        raise ToolArchiveIndexError("Unsupported tool archive index schema")
    expected_digest = index.get("index_sha256")
    if not isinstance(expected_digest, str) or not _SHA256_RE.fullmatch(expected_digest):
        raise ToolArchiveIndexError("Tool archive index has no valid self digest")
    if index_digest(index) != expected_digest:
        raise ToolArchiveIndexError("Tool archive index self digest mismatch")

    archives = index.get("archives")
    members = index.get("members")
    non_tool_members = index.get("non_tool_members")
    api_indexes = index.get("api_indexes")
    if (
        not isinstance(archives, list)
        or not isinstance(members, dict)
        or not isinstance(non_tool_members, dict)
    ):
        raise ToolArchiveIndexError("Tool archive index archives/members are malformed")
    if not isinstance(api_indexes, dict):
        raise ToolArchiveIndexError("Tool archive API indexes are malformed")
    archive_by_path: dict[str, dict[str, Any]] = {}
    member_counts: dict[str, int] = {}
    for archive in archives:
        if not isinstance(archive, dict):
            raise ToolArchiveIndexError("Malformed archive record")
        path = str(archive.get("path", ""))
        _safe_relative_path(path, label="archive")
        if path in archive_by_path:
            raise ToolArchiveIndexError(f"Duplicate archive record: {path}")
        if archive.get("format") not in {"tar.gz", "tar.zst"}:
            raise ToolArchiveIndexError(f"Unsupported archive format in index: {path}")
        if archive.get("logical_root") not in {
            f"{root}/" for root in EXPECTED_TOOL_ROOTS
        }:
            raise ToolArchiveIndexError(f"Undeclared logical root for archive: {path}")
        archive_by_path[path] = archive
        member_counts[path] = 0

    for logical_path, member in {**members, **non_tool_members}.items():
        _safe_relative_path(str(logical_path), label="logical member")
        if not isinstance(member, dict):
            raise ToolArchiveIndexError(f"Malformed member record: {logical_path}")
        if logical_path in members:
            _logical_root(PurePosixPath(logical_path))
        elif member.get("reason") != "declared_archive_metadata":
            raise ToolArchiveIndexError(
                f"Non-tool archive member has no declared reason: {logical_path}"
            )
        archive_path = str(member.get("archive_path", ""))
        if archive_path not in archive_by_path:
            raise ToolArchiveIndexError(
                f"Member references undeclared archive: {logical_path}"
            )
        _safe_relative_path(str(member.get("member_path", "")), label="archive member")
        if not isinstance(member.get("size_bytes"), int) or member["size_bytes"] < 0:
            raise ToolArchiveIndexError(f"Invalid member size: {logical_path}")
        if not _SHA256_RE.fullmatch(str(member.get("sha256", ""))):
            raise ToolArchiveIndexError(f"Invalid member digest: {logical_path}")
        member_counts[archive_path] += 1

    for path, archive in archive_by_path.items():
        if member_counts[path] != archive.get("member_count"):
            raise ToolArchiveIndexError(f"Archive member count mismatch: {path}")
        tool_count = sum(
            member["archive_path"] == path for member in members.values()
        )
        non_tool_count = sum(
            member["archive_path"] == path for member in non_tool_members.values()
        )
        if tool_count != archive.get("tool_member_count"):
            raise ToolArchiveIndexError(f"Archive tool-member count mismatch: {path}")
        if non_tool_count != archive.get("non_tool_member_count"):
            raise ToolArchiveIndexError(
                f"Archive non-tool-member count mismatch: {path}"
            )
    if archive_set_digest(archives) != index.get("archive_set_sha256"):
        raise ToolArchiveIndexError("Archive-set digest mismatch")
    if tool_profile is not None:
        if index.get("tool_profile_source_tree_sha256") != tool_profile.get(
            "source_tree_sha256"
        ):
            raise ToolArchiveIndexError("Tool profile digest does not match archive index")
        declared = {
            str(item["path"]): str(item["sha256"])
            for item in tool_profile.get("shared_tool_archives", [])
        }
        indexed = {
            str(item["path"]): str(item["sha256"])
            for item in archives
        }
        if indexed != declared:
            raise ToolArchiveIndexError(
                "Archive index does not cover the exact tool-profile archive set"
            )
        if set(non_tool_members) != set(
            tool_profile.get("non_tool_archive_members", [])
        ):
            raise ToolArchiveIndexError(
                "Archive index non-tool members do not match the tool profile"
            )
    if require_archive_files is not None:
        for path, archive in archive_by_path.items():
            candidate = require_archive_files / path
            if not candidate.is_file():
                raise ToolArchiveIndexError(f"Indexed archive is missing: {path}")
            if sha256_file(candidate) != archive["sha256"]:
                raise ToolArchiveIndexError(f"Indexed archive hash mismatch: {path}")


def load_tool_archive_index(
    path: Path,
    *,
    tool_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        index = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ToolArchiveIndexError(f"Invalid tool archive index at {path}: {error}") from error
    if not isinstance(index, dict):
        raise ToolArchiveIndexError(f"Expected a JSON object at {path}")
    validate_tool_archive_index(index, tool_profile=tool_profile)
    return index


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        release = BVBRelease.open(args.release_dir)
        index = build_tool_archive_index(release)
        output = args.output or release.root / INDEX_PATH
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(stable_json_bytes(index))
    except (ReleaseError, OSError) as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                "output": str(output),
                "index_sha256": index["index_sha256"],
                "archive_set_sha256": index["archive_set_sha256"],
                "members": len(index["members"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build a reusable Daytona snapshot containing the BVB MCP and shared tools."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import shutil
import subprocess
from pathlib import Path

from .adapter import _DAYTONA_MCP_ENV, TEMPLATE_DIR, runtime_template_sha256, snapshot_revision_label
from .config import DATASET_REVISION, DEFAULT_RELEASE_DIR
from .prerequisites import validate_release
from .release import BVBRelease, ReleaseError, sha256_file
from .tool_archive_index import stable_json_bytes


def _snapshot_identity(
    release: BVBRelease,
    *,
    release_revision: str,
    cache_max_bytes: int,
) -> dict[str, object]:
    index = release.tool_archive_index(require_archive_files=True)
    return {
        "schema_version": "bvb-daytona-snapshot/v2",
        "release_revision": release_revision,
        "mcp_tree_sha256": str(release.tool_profile["source_tree_sha256"]),
        "archive_index_sha256": str(index["index_sha256"]),
        "archive_set_sha256": str(index["archive_set_sha256"]),
        "runtime_sha256": runtime_template_sha256(),
        "cache_max_bytes": cache_max_bytes,
    }


def snapshot_name(
    release: BVBRelease,
    *,
    release_revision: str = DATASET_REVISION,
) -> str:
    """Derive a readable snapshot name from all immutable lazy-tool inputs."""
    index = release.tool_archive_index()
    revision = snapshot_revision_label(release_revision)
    mcp = str(release.tool_profile["source_tree_sha256"])[:8]
    index_digest = str(index["index_sha256"])[:8]
    archives = str(index["archive_set_sha256"])[:8]
    runtime = runtime_template_sha256()[:8]
    return f"bvb-r{revision}-m{mcp}-i{index_digest}-a{archives}-t{runtime}"


_EXPECTED_MCP_TOOLS = {
    "copy_cik_lookup",
    "get_company_data",
    "get_company_facts",
    "get_data_description",
    "get_equity_risk_premium",
    "get_filing",
    "get_frames",
    "get_industry_betas",
    "get_logo_file",
    "get_submissions",
    "list_available_data",
    "list_available_filings",
    "list_available_logos",
    "list_tickers",
    "search_logos",
}


def _python_stdin_command(source: str, *, user: str | None = None) -> str:
    encoded = base64.b64encode(source.encode()).decode()
    interpreter = (
        f"sudo -n -u {user} env PATH=/opt/agent-python/bin:/usr/local/sbin:"
        "/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin python3"
        if user
        else "python3"
    )
    return f"printf %s '{encoded}' | base64 -d | {interpreter}"


def _judge_runtime_probe_source() -> str:
    return f"""
import json
import subprocess
import sys

import openpyxl
import pandas
import pdfplumber
import pptx
import pypdf
import reportlab

messages = [
    {{
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {{
            "protocolVersion": "2025-06-18",
            "capabilities": {{}},
            "clientInfo": {{"name": "bvb-preflight", "version": "1"}},
        }},
    }},
    {{"jsonrpc": "2.0", "method": "notifications/initialized", "params": {{}}}},
    {{"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {{}}}},
]
process = subprocess.Popen(
    ["/usr/bin/env", *{list(_DAYTONA_MCP_ENV)!r}, "/usr/bin/mcp-server"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
)
stdout, stderr = process.communicate(
    "".join(json.dumps(message) + "\\n" for message in messages),
    timeout=60,
)
if process.returncode:
    raise RuntimeError(
        f"MCP server exited {{process.returncode}}: {{stderr.strip() or stdout.strip()}}"
    )
responses = [json.loads(line) for line in stdout.splitlines() if line.strip()]
tools_response = next(
    (response for response in responses if response.get("id") == 2),
    None,
)
if tools_response is None:
    raise RuntimeError(f"MCP tools/list returned no response: {{stdout}} {{stderr}}")
tool_names = sorted(
    tool["name"] for tool in tools_response.get("result", {{}}).get("tools", [])
)
expected = {sorted(_EXPECTED_MCP_TOOLS)!r}
missing = sorted(set(expected) - set(tool_names))
if missing:
    raise RuntimeError(f"MCP tools missing from inner judge: {{missing}}")
print(json.dumps({{"python": sys.executable, "mcp_tools": tool_names}}))
"""


async def _probe_snapshot_sandbox(
    sandbox,
    *,
    snapshot: str,
    expected_identity: dict[str, object],
) -> dict[str, object]:
    identity_source = """
import json
from pathlib import Path
print(json.dumps(json.loads(
    Path("/opt/mcp-server/data/bvb-snapshot-identity.json").read_text()
)))
"""
    identity_result = await sandbox.process.exec(
        _python_stdin_command(identity_source),
        timeout=60,
    )
    if identity_result.exit_code != 0:
        raise ReleaseError(
            "Unable to read Daytona snapshot identity: "
            f"{identity_result.result}"
        )
    observed_identity = json.loads(identity_result.result.strip())
    if observed_identity != expected_identity:
        raise ReleaseError(
            f"Daytona snapshot {snapshot!r} has an incompatible runtime identity"
        )

    probe_result = await sandbox.process.exec(
        _python_stdin_command(
            _judge_runtime_probe_source(),
            user="sandbox",
        ),
        timeout=120,
    )
    if probe_result.exit_code != 0:
        raise ReleaseError(
            "Daytona inner-judge preflight failed: "
            f"{probe_result.result}"
        )
    probe = json.loads(probe_result.result.strip())
    if not str(probe.get("python", "")).startswith("/opt/agent-python/"):
        raise ReleaseError(
            "Daytona inner judge resolves python3 outside /opt/agent-python: "
            f"{probe.get('python')}"
        )
    return {
        "snapshot": snapshot,
        "identity": observed_identity,
        "python": probe["python"],
        "mcp_tools": probe["mcp_tools"],
    }


async def _validate_snapshot_runtime(
    snapshot: str,
    expected_identity: dict[str, object],
) -> dict[str, object]:
    from daytona import AsyncDaytona, CreateSandboxFromSnapshotParams

    try:
        async with AsyncDaytona() as daytona:
            sandbox = await daytona.create(
                CreateSandboxFromSnapshotParams(snapshot=snapshot, ephemeral=True),
                timeout=0,
            )
            try:
                return await _probe_snapshot_sandbox(
                    sandbox,
                    snapshot=snapshot,
                    expected_identity=expected_identity,
                )
            finally:
                await sandbox.delete()
    except ReleaseError:
        raise
    except Exception as error:
        raise ReleaseError(
            f"Unable to preflight Daytona snapshot {snapshot!r}: {error}"
        ) from error


def validate_snapshot_runtime(
    release: BVBRelease,
    snapshot: str,
    *,
    release_revision: str = DATASET_REVISION,
    cache_max_bytes: int = 2 * 1024 * 1024 * 1024,
) -> dict[str, object]:
    """Fail before paid judging unless the snapshot exposes inspection and MCP tools."""
    expected_identity = _snapshot_identity(
        release,
        release_revision=release_revision,
        cache_max_bytes=cache_max_bytes,
    )
    expected_name = snapshot_name(release, release_revision=release_revision)
    if snapshot != expected_name:
        raise ReleaseError(
            f"Daytona snapshot must match the current runtime identity: {expected_name}"
        )
    return asyncio.run(_validate_snapshot_runtime(snapshot, expected_identity))


def _context_dockerfile(
    *,
    identity: dict[str, object],
    include_archives: bool,
    include_index: bool,
) -> str:
    """Return the BVB runtime Dockerfile with lazy archive configuration."""
    environment = TEMPLATE_DIR / "environment"
    dockerfile = (environment / "Dockerfile").read_text(encoding="utf-8")
    dockerfile = "\n".join(
        line for line in dockerfile.splitlines() if "COPY --chown=agent:agent input/" not in line
    )
    archive_copy = (
        "COPY shared-tools/ /opt/mcp-server/data/shared-tools/\n"
        if include_archives
        else ""
    )
    index_copy = (
        "COPY tool-archive-index.json /opt/mcp-server/data/tool-archive-index.json\n"
        if include_index
        else ""
    )
    index_permission = (
        " /opt/mcp-server/data/tool-archive-index.json"
        if include_index
        else ""
    )
    return (
        dockerfile
        + f"""

USER root
COPY bvb-snapshot-identity.json /opt/mcp-server/data/bvb-snapshot-identity.json
{index_copy}{archive_copy}RUN mkdir -p /var/cache/bvb-tools /opt/mcp-server/data/shared-tools && \
    chown -R environment:environment /var/cache/bvb-tools && \
    chmod 700 /var/cache/bvb-tools && \
    chmod -R a-w /opt/mcp-server/data/shared-tools && \
    chmod a+rX /opt/mcp-server/data /opt/mcp-server/data/shared-tools \
      /opt/mcp-server/data/bvb-snapshot-identity.json{index_permission}
ENV BVB_TOOL_ARCHIVE_INDEX=/opt/mcp-server/data/tool-archive-index.json
ENV BVB_TOOL_ARCHIVE_ROOT=/opt/mcp-server/data
ENV BVB_TOOL_CACHE_ROOT=/var/cache/bvb-tools
ENV BVB_TOOL_CACHE_MAX_BYTES={identity["cache_max_bytes"]}
USER agent
"""
    )


def _write_snapshot_metadata(
    release: BVBRelease,
    context_dir: Path,
    identity: dict[str, object],
    *,
    include_index: bool,
) -> None:
    if include_index:
        shutil.copy2(
            release.tool_archive_index_path,
            context_dir / "tool-archive-index.json",
        )
    (context_dir / "bvb-snapshot-identity.json").write_bytes(
        stable_json_bytes(identity)
    )


def _link_or_copy(source: str, destination: str) -> str:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return destination


def prepare_snapshot_context(
    release: BVBRelease,
    context_dir: Path,
    *,
    release_revision: str = DATASET_REVISION,
    cache_max_bytes: int = 2 * 1024 * 1024 * 1024,
) -> Path:
    """Build a compressed lazy-tool Docker context without extracting archives."""
    context_dir = context_dir.resolve()
    if context_dir.exists():
        shutil.rmtree(context_dir)
    context_dir.mkdir(parents=True)
    environment = TEMPLATE_DIR / "environment"
    shutil.copytree(release.mcp_root, context_dir / "mcp-server")
    shutil.copy2(environment / "agent-python-requirements.txt", context_dir)
    shutil.copy2(environment / ".dockerignore", context_dir)
    identity = _snapshot_identity(
        release,
        release_revision=release_revision,
        cache_max_bytes=cache_max_bytes,
    )
    _write_snapshot_metadata(
        release,
        context_dir,
        identity,
        include_index=True,
    )
    shutil.copytree(
        release.shared_tools_dir,
        context_dir / "shared-tools",
        copy_function=_link_or_copy,
    )
    dockerfile = _context_dockerfile(
        identity=identity,
        include_archives=True,
        include_index=True,
    )
    (context_dir / "Dockerfile").write_text(dockerfile, encoding="utf-8")
    return context_dir


def prepare_seed_context(
    release: BVBRelease,
    context_dir: Path,
    *,
    release_revision: str = DATASET_REVISION,
    cache_max_bytes: int = 2 * 1024 * 1024 * 1024,
) -> Path:
    """Prepare a small base image context before streaming archives to Daytona."""
    context_dir = context_dir.resolve()
    if context_dir.exists():
        shutil.rmtree(context_dir)
    context_dir.mkdir(parents=True)
    environment = TEMPLATE_DIR / "environment"
    shutil.copytree(release.mcp_root, context_dir / "mcp-server")
    shutil.copy2(environment / "agent-python-requirements.txt", context_dir)
    shutil.copy2(environment / ".dockerignore", context_dir)
    identity = _snapshot_identity(
        release,
        release_revision=release_revision,
        cache_max_bytes=cache_max_bytes,
    )
    _write_snapshot_metadata(
        release,
        context_dir,
        identity,
        include_index=False,
    )
    dockerfile = _context_dockerfile(
        identity=identity,
        include_archives=False,
        include_index=False,
    )
    # Archive streaming needs root-owned writes beneath immutable /opt.
    (context_dir / "Dockerfile").write_text(
        dockerfile + "\nUSER root\n",
        encoding="utf-8",
    )
    return context_dir


async def _upload_verified(
    sandbox,
    *,
    source: Path,
    destination: str,
    expected_sha256: str,
    attempts: int = 5,
) -> None:
    """Retry transient Daytona uploads and accept only the expected bytes."""
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            await sandbox.fs.upload_file_stream(
                str(source),
                destination,
                timeout=0,
            )
            last_error = None
        except Exception as error:  # Daytona wraps several transport errors.
            last_error = error
        try:
            check = await sandbox.process.exec(
                f"test -f {destination} && "
                f"test \"$(sha256sum {destination} | cut -d' ' -f1)\" = "
                f"\"{expected_sha256}\"",
                timeout=3600,
            )
            if check.exit_code == 0:
                return
            await sandbox.process.exec(f"rm -f {destination}", timeout=60)
        except Exception as error:
            last_error = error
        if attempt < attempts:
            delay = min(30, 2 ** attempt)
            print(
                f"Retrying upload for {source.name} after attempt {attempt}/{attempts}"
            )
            await asyncio.sleep(delay)
    detail = f": {last_error}" if last_error else ""
    raise ReleaseError(
        f"Unable to upload and verify {source.name} after {attempts} attempts{detail}"
    )


async def _create_snapshot_from_seed(
    release: BVBRelease,
    seed_context: Path,
    name: str,
    disk_gb: int,
    identity_sha256: str,
    index_file_sha256: str,
) -> None:
    """Avoid build-context timeouts by streaming compressed archives into a sandbox."""
    from daytona import (
        AsyncDaytona,
        CreateSandboxFromImageParams,
        CreateSandboxFromSnapshotParams,
        Image,
        Resources,
    )
    from daytona import DaytonaNotFoundError

    async with AsyncDaytona() as daytona:
        try:
            existing = await daytona.snapshot.get(name)
        except DaytonaNotFoundError:
            existing = None
        if existing is not None:
            if str(existing.state).upper().endswith("ACTIVE"):
                verifier = await daytona.create(
                    CreateSandboxFromSnapshotParams(
                        snapshot=name,
                        ephemeral=True,
                    ),
                    timeout=0,
                )
                try:
                    result = await verifier.process.exec(
                        "sha256sum /opt/mcp-server/data/bvb-snapshot-identity.json",
                        timeout=60,
                    )
                    observed = result.result.strip().split(maxsplit=1)[0]
                    if result.exit_code != 0 or observed != identity_sha256:
                        raise ReleaseError(
                            f"Daytona snapshot {name!r} has an incompatible embedded identity"
                        )
                finally:
                    await verifier.delete()
                print(f"Reusing verified active Daytona snapshot: {name}")
                return
            raise ReleaseError(f"Daytona snapshot {name!r} already exists in state {existing.state}")

        sandbox = await daytona.create(
            CreateSandboxFromImageParams(
                image=Image.from_dockerfile(seed_context / "Dockerfile"),
                resources=Resources(cpu=4, memory=8, disk=disk_gb),
                ephemeral=True,
            ),
            timeout=0,
            on_snapshot_create_logs=print,
        )
        try:
            setup = await sandbox.process.exec(
                "mkdir -p /opt/mcp-server/data/shared-tools",
                timeout=60,
            )
            if setup.exit_code != 0:
                raise ReleaseError(f"Unable to initialize Daytona tool paths: {setup.result}")
            print("Uploading canonical tool archive index")
            await _upload_verified(
                sandbox,
                source=release.tool_archive_index_path,
                destination="/opt/mcp-server/data/tool-archive-index.json",
                expected_sha256=index_file_sha256,
            )
            archives = release.tool_profile["shared_tool_archives"]
            for position, entry in enumerate(archives, start=1):
                source = release.root / entry["path"]
                destination = f"/opt/mcp-server/data/shared-tools/{source.name}"
                print(f"Uploading shared-tool archive {position}/{len(archives)}: {source.name}")
                await _upload_verified(
                    sandbox,
                    source=source,
                    destination=destination,
                    expected_sha256=str(entry["sha256"]),
                )
            archive_checks = " && ".join(
                f"test \"$(sha256sum /opt/mcp-server/data/shared-tools/{Path(entry['path']).name} "
                f"| cut -d' ' -f1)\" = \"{entry['sha256']}\""
                for entry in archives
            )
            finalization = await sandbox.process.exec(
                "set -eu; "
                f"{archive_checks}; "
                "chown -R root:root /opt/mcp-server/data/shared-tools; "
                "chmod -R a-w /opt/mcp-server/data/shared-tools; "
                "chmod -R a+rX /opt/mcp-server/data/shared-tools; "
                f"test \"$(sha256sum /opt/mcp-server/data/tool-archive-index.json "
                f"| cut -d' ' -f1)\" = \"{index_file_sha256}\"; "
                "chmod a-w,a+r /opt/mcp-server/data/tool-archive-index.json; "
                f"test \"$(sha256sum /opt/mcp-server/data/bvb-snapshot-identity.json "
                f"| cut -d' ' -f1)\" = \"{identity_sha256}\"",
                timeout=3600,
            )
            if finalization.exit_code != 0:
                raise ReleaseError(
                    "Unable to verify compressed BVB archives in Daytona: "
                    f"{finalization.result}"
                )
            await sandbox.create_snapshot(name, timeout=0)
        finally:
            await sandbox.delete()


def create_snapshot(
    release: BVBRelease,
    context_dir: Path,
    name: str,
    disk_gb: int,
    *,
    release_revision: str = DATASET_REVISION,
    cache_max_bytes: int = 2 * 1024 * 1024 * 1024,
) -> None:
    """Seed and snapshot BVB tool data using resilient archive streaming."""
    if not os.environ.get("DAYTONA_API_KEY"):
        raise ReleaseError("DAYTONA_API_KEY is required to create a Daytona snapshot")
    seed_context = prepare_seed_context(
        release,
        context_dir.with_name(context_dir.name + "-seed"),
        release_revision=release_revision,
        cache_max_bytes=cache_max_bytes,
    )
    identity_sha256 = sha256_file(seed_context / "bvb-snapshot-identity.json")
    index_file_sha256 = sha256_file(release.tool_archive_index_path)
    asyncio.run(
        _create_snapshot_from_seed(
            release,
            seed_context,
            name,
            disk_gb,
            identity_sha256,
            index_file_sha256,
        )
    )
    validate_snapshot_runtime(
        release,
        name,
        release_revision=release_revision,
        cache_max_bytes=cache_max_bytes,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE_DIR)
    parser.add_argument("--context-dir", type=Path, required=True)
    parser.add_argument("--snapshot-name")
    parser.add_argument(
        "--snapshot-disk-gb",
        type=int,
        default=20,
        help="Daytona sandbox disk for seeding the snapshot (default: 20 GB account limit).",
    )
    parser.add_argument(
        "--release-revision",
        default=DATASET_REVISION,
        help="Dataset revision (tag or commit) embedded in snapshot identity.",
    )
    parser.add_argument(
        "--cache-max-bytes",
        type=int,
        default=2 * 1024 * 1024 * 1024,
        help="Per-sandbox bounded lazy extraction cache (default: 2 GiB).",
    )
    parser.add_argument("--create", action="store_true", help="Create the Daytona snapshot after preparing context.")
    args = parser.parse_args(argv)
    try:
        release, _ = validate_release(args.release_dir, require_tools=True)
        context = prepare_snapshot_context(
            release,
            args.context_dir,
            release_revision=args.release_revision,
            cache_max_bytes=args.cache_max_bytes,
        )
        expected_name = snapshot_name(
            release,
            release_revision=args.release_revision,
        )
        if args.snapshot_name and args.snapshot_name != expected_name:
            raise ReleaseError(
                f"Snapshot name must encode the canonical inputs: {expected_name}"
            )
        name = expected_name
        if args.create:
            create_snapshot(
                release,
                context,
                name,
                args.snapshot_disk_gb,
                release_revision=args.release_revision,
                cache_max_bytes=args.cache_max_bytes,
            )
    except (ReleaseError, OSError, subprocess.CalledProcessError) as error:
        parser.error(str(error))
    print(json.dumps({"context_dir": str(context), "snapshot": name, "created": args.create}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

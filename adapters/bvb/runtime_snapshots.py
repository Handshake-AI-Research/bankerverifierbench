"""Derived Daytona runtimes used by the paper's model-comparison conditions.

The base snapshot from ``adapters.bvb.daytona`` runs the pinned native Gandalf.
The Gandalf model comparisons ran on a derived ``llm`` runtime: LiteLLM 1.104.0
for newer model IDs, the upstream OpenHands reasoning-capability check
backported into SDK 1.14.0, and group-readable exported logos. The RewardKit
comparisons ran on a derived ``rewardkit`` runtime with RewardKit 0.1.2 and
Codex 0.160.1. Grader prompts, permissions and evidence are unchanged.

Each derived snapshot is named by a hash of its identity, which the runners
verify inside the sandbox before any model call.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

from .config import DEFAULT_RELEASE_DIR
from .daytona import snapshot_name
from .prerequisites import validate_release
from .release import BVBRelease, ReleaseError

TEMPLATE_DIR = Path(__file__).resolve().parent / "template"
LOGO_PATCH = TEMPLATE_DIR / "_environment-rewardkit" / "logo-export-permissions.patch"
SDK_PATCH = TEMPLATE_DIR / "_environment-gandalf" / "model-capabilities.patch"
CODEX_WRAPPER = TEMPLATE_DIR / "_environment-rewardkit" / "codex-wrapper.py"
GANDALF_PYTHON = "/opt/uv-tools/gandalf-the-grader/bin/python"

GANDALF_PACKAGES = {
    "gandalf-the-grader": "1.0.0",
    "openhands-sdk": "1.14.0",
    "openhands-tools": "1.16.0",
    "litellm": "1.104.0",
}
LOGO_EXPORT_PERMISSIONS = {
    "path": "/opt/mcp-server/tools/company_brand_assets.py",
    "original_sha256": "6095132e84c8cec05adaaefd8d2e199dc7260c24edd29db29efa62b7ed6b6c2c",
    "patched_sha256": "640db9f18bdf196b2ed0779453f6a9b7e105d649b379fb137ee31f31a57bccec",
    "scope": "Exported logo receives group read only (0640); original MCP/cache directory permissions unchanged",
}
SDK_REASONING_CAPABILITY_BACKPORT = {
    "path": "/opt/uv-tools/gandalf-the-grader/lib/python3.12/site-packages/openhands/sdk/llm/utils/model_features.py",
    "original_sha256": "5576e21159c84bbc7dd7fd36482c25533224fbd17d05cf64ce57d0764713ce66",
    "patched_sha256": "bb143ad7d9658cedc5f7858449e299a775853c0c2df00684cb23ed4dcdce1897",
    "upstream_reference": "OpenHands SDK 1.15/1.16 model_features.py: LiteLLM-supported parameter detection replaces static reasoning-model list",
    "scope": "Only reasoning-effort capability detection; no model aliases, context-window changes, grader prompts, or sandbox changes",
    "reason": "SDK 1.15/1.16 pins LiteLLM 1.80.10, while the requested models require 1.104.0. Backport the upstream gate in the existing compatible SDK 1.14 runtime.",
}
REWARDKIT_VERSION = "0.1.2"
CODEX_VERSION = "0.160.1"
TOMLI_W_VERSION = "1.2.0"

KINDS = {
    "gandalf-llm": ("llm", "/opt/bvb-model-runtime-identity.json"),
    "rewardkit": ("rewardkit", "/opt/bvb-rewardkit-identity.json"),
}


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def plan(kind: str, release: BVBRelease | None = None, *, base_snapshot: str | None = None) -> dict:
    """Return ``{"snapshot", "identity"}`` for a derived runtime of the base snapshot."""
    if kind not in KINDS:
        raise ReleaseError(f"Unknown runtime kind {kind!r}; expected one of {sorted(KINDS)}")
    if base_snapshot is None:
        if release is None:
            raise ReleaseError("A release or an explicit base snapshot is required")
        base_snapshot = snapshot_name(release, release_revision=release.revision)
    if kind == "gandalf-llm":
        identity = {
            "base_snapshot": base_snapshot,
            "packages": dict(GANDALF_PACKAGES),
            "logo_export_permissions": dict(LOGO_EXPORT_PERMISSIONS),
            "sdk_reasoning_capability_backport": dict(SDK_REASONING_CAPABILITY_BACKPORT),
        }
    else:
        identity = {
            "rewardkit_version": REWARDKIT_VERSION,
            "codex_version": CODEX_VERSION,
            "codex_wrapper_sha256": _sha256(CODEX_WRAPPER.read_bytes()),
            "base_snapshot": base_snapshot,
            "logo_export_permissions": dict(LOGO_EXPORT_PERMISSIONS),
        }
    digest = _sha256(json.dumps(identity, sort_keys=True).encode())[:8]
    return {"snapshot": f"{base_snapshot}-{KINDS[kind][0]}-{digest}", "identity": identity}


def patched_bytes(original: bytes, patch: Path, fix: dict) -> bytes:
    """Apply ``patch`` to ``original`` and require both recorded hashes."""
    if _sha256(original) != fix["original_sha256"]:
        raise ReleaseError(f"Unexpected original source for {fix['path']}")
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "source.py"
        target.write_bytes(original)
        subprocess.run(["patch", "--silent", "--forward", str(target), str(patch)], check=True)
        result = target.read_bytes()
    if _sha256(result) != fix["patched_sha256"]:
        raise ReleaseError(f"Patched source for {fix['path']} differs from the recorded runtime")
    return result


async def _exec(sandbox, command: str, timeout: int) -> str:
    result = await sandbox.process.exec(command, timeout=timeout)
    if result.exit_code:
        raise ReleaseError(f"Runtime build step failed ({result.exit_code}): {command}\n{result.result[-2000:]}")
    return result.result


async def _replace_file(sandbox, fix: dict, payload: bytes, *, owner: str | None) -> None:
    await sandbox.fs.upload_file(payload, fix["path"])
    chown = f"chown {owner} {fix['path']} && " if owner else ""
    output = await _exec(sandbox, f"{chown}chmod 644 {fix['path']} && sha256sum {fix['path']}", 30)
    if output.split()[0] != fix["patched_sha256"]:
        raise ReleaseError(f"Uploaded source for {fix['path']} differs from the recorded runtime")


async def _install_gandalf_llm(sandbox, release: BVBRelease) -> None:
    await _exec(
        sandbox,
        f"uv pip install --python {GANDALF_PYTHON} 'litellm=={GANDALF_PACKAGES['litellm']}' && chmod -R a+rX /opt/uv-tools",
        300,
    )
    source = (
        "import importlib.metadata as m, json; "
        f"print(json.dumps({{p: m.version(p) for p in {sorted(GANDALF_PACKAGES)!r}}}))"
    )
    versions = json.loads(await _exec(sandbox, f"{GANDALF_PYTHON} -c {json.dumps(source)}", 60))
    if versions != GANDALF_PACKAGES:
        raise ReleaseError(f"Gandalf runtime package versions differ: {versions}")
    logo = (release.mcp_root / "tools" / "company_brand_assets.py").read_bytes()
    await _replace_file(
        sandbox, LOGO_EXPORT_PERMISSIONS, patched_bytes(logo, LOGO_PATCH, LOGO_EXPORT_PERMISSIONS),
        owner="environment:environment",
    )
    sdk = await sandbox.fs.download_file(SDK_REASONING_CAPABILITY_BACKPORT["path"])
    await _replace_file(
        sandbox, SDK_REASONING_CAPABILITY_BACKPORT,
        patched_bytes(sdk, SDK_PATCH, SDK_REASONING_CAPABILITY_BACKPORT), owner=None,
    )


async def _install_rewardkit(sandbox, release: BVBRelease) -> None:
    await sandbox.fs.upload_file(CODEX_WRAPPER.read_bytes(), "/tmp/codex-wrapper.py")
    await _exec(
        sandbox,
        "UV_TOOL_DIR=/opt/uv-tools UV_TOOL_BIN_DIR=/usr/local/bin UV_PYTHON_INSTALL_DIR=/opt/uv-python "
        f"uv tool install 'harbor-rewardkit=={REWARDKIT_VERSION}' && "
        f"uv pip install --python /opt/agent-python/bin/python3 tomli-w=={TOMLI_W_VERSION} && "
        "chmod -R a+rX /opt/uv-tools /opt/uv-python && "
        f"npm install -g @openai/codex@{CODEX_VERSION} && "
        "mv /usr/local/bin/codex /usr/local/bin/codex-native && "
        "cp /tmp/codex-wrapper.py /usr/local/bin/codex && chmod 755 /usr/local/bin/codex",
        300,
    )
    env = "sudo -n -u sandbox env HOME=/tmp PATH=/opt/agent-python/bin:/usr/local/bin:/usr/bin:/bin"
    await _exec(sandbox, f"{env} codex --version && {env} rewardkit --help", 60)
    logo = (release.mcp_root / "tools" / "company_brand_assets.py").read_bytes()
    await _replace_file(
        sandbox, LOGO_EXPORT_PERMISSIONS, patched_bytes(logo, LOGO_PATCH, LOGO_EXPORT_PERMISSIONS),
        owner="environment:environment",
    )


async def _build(kind: str, release: BVBRelease, runtime: dict) -> None:
    from daytona import AsyncDaytona, CreateSandboxFromSnapshotParams

    async with AsyncDaytona() as daytona:
        sandbox = await daytona.create(
            CreateSandboxFromSnapshotParams(snapshot=runtime["identity"]["base_snapshot"], ephemeral=True),
            timeout=120,
        )
        try:
            await (_install_gandalf_llm if kind == "gandalf-llm" else _install_rewardkit)(sandbox, release)
            identity = json.dumps(runtime["identity"], indent=2).encode()
            await sandbox.fs.upload_file(identity, KINDS[kind][1])
            await sandbox.create_snapshot(runtime["snapshot"], timeout=0)
        finally:
            await sandbox.delete()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kind", choices=sorted(KINDS), required=True)
    parser.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE_DIR)
    parser.add_argument("--create", action="store_true", help="Build and publish the derived Daytona snapshot.")
    args = parser.parse_args(argv)
    try:
        release, _ = validate_release(args.release_dir, require_tools=True)
        runtime = plan(args.kind, release)
        if args.create:
            asyncio.run(_build(args.kind, release, runtime))
    except (ReleaseError, OSError, subprocess.CalledProcessError) as error:
        parser.error(str(error))
    print(json.dumps(dict(runtime, created=args.create), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

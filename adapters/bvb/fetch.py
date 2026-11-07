"""Download the pinned BankerVerifierBench release and its shared-tool archives.

The dataset comes from DATASET_REPOSITORY at DATASET_REVISION and must match
DATASET_MANIFEST_SHA256. The 19 tool
archives are byte-identical to public BankerToolBench and are downloaded from
the repository and revision named in the release's manifests/shared-tools.json,
then checked against the hashes recorded there.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .config import DATASET_REPOSITORY, DATASET_REVISION, DEFAULT_RELEASE_DIR, SHARED_TOOLS_DIR
from .release import BVBRelease, ReleaseError, ValidationResult


def _download(repo_id: str, revision: str, local_dir: Path, token: str | None,
              allow_patterns: list[str] | None = None) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:  # pragma: no cover - dependency is declared
        raise ReleaseError("huggingface-hub is required; run `uv sync` first") from error
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            allow_patterns=allow_patterns,
            token=token,
            local_dir=str(local_dir),
        )
    except Exception as error:  # Hugging Face errors vary by installed version.
        raise ReleaseError(
            f"Unable to download {repo_id}@{revision}. If the repository is private, "
            "authenticate with `hf auth login` or set HF_TOKEN; the token value is never logged."
        ) from error


def fetch_release(
    *,
    output_dir: Path = DEFAULT_RELEASE_DIR,
    with_tools: bool = False,
    token: str | None = None,
) -> tuple[BVBRelease, ValidationResult]:
    """Download the release (and optionally its tool archives) and validate every file."""
    token = token or os.environ.get("HF_TOKEN")
    output_dir = output_dir.resolve()
    _download(DATASET_REPOSITORY, DATASET_REVISION, output_dir, token)
    release = BVBRelease.open(output_dir)
    if with_tools:
        tools = release.shared_tools()
        _download(tools["repository"], tools["revision"], output_dir, token,
                  allow_patterns=[f"{SHARED_TOOLS_DIR}/*"])
    return release, release.validate_hashes(require_all_files=with_tools)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--with-tools",
        action="store_true",
        help="Also download the 19 BankerToolBench shared-tool archives (5.6 GB) needed to run verifiers.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RELEASE_DIR,
        help=f"Local release directory (default: {DEFAULT_RELEASE_DIR}).",
    )
    args = parser.parse_args(argv)
    try:
        release, result = fetch_release(output_dir=args.output_dir, with_tools=args.with_tools)
    except ReleaseError as error:
        parser.error(str(error))
    tools = release.shared_tools()
    print(
        json.dumps(
            {
                "dataset": DATASET_REPOSITORY,
                "revision": DATASET_REVISION,
                "release_root": str(release.root),
                "shared_tools": f"{tools['repository']}@{tools['revision']}",
                "with_tools": args.with_tools,
                "tasks": result.task_count,
                "adjudicated_tasks": result.adjudicated_task_count,
                "eligible_criteria": result.eligible_criteria_count,
                "missing_tool_archives": len(
                    [path for path in result.missing_files if path.startswith(f"{SHARED_TOOLS_DIR}/")]
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

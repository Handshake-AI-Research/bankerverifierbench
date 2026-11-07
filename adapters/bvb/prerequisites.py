"""BankerVerifierBench preflight checks shared by public operator commands."""

from __future__ import annotations

from pathlib import Path

from .release import BVBRelease, ValidationResult


def validate_release(
    release_dir: Path,
    *,
    require_tools: bool = False,
) -> tuple[BVBRelease, ValidationResult]:
    """Open and verify a release downloaded at the pinned revision."""
    release = BVBRelease.open(release_dir)
    release.assert_pinned_revision()
    return release, release.validate_hashes(require_all_files=require_tools)

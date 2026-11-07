"""Constants and path helpers for BankerVerifierBench (BVB)."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATASET_REPOSITORY = "handshake-ai-research/BankerVerifierBench"
DATASET_REVISION = "e8068910360bcb54459a18c97cfc8d1548da37c1"
# Commit that tag v1.0 names. SHA-256 of manifests/files.sha256 at that commit.
DATASET_MANIFEST_SHA256 = "c72e25f07964d910803c09ce0c839e1b65ca2d8fc29c74aace8a4ff079e7143d"
DATASET_TYPE = "dataset"
RELEASE_NAME = "bvb"
RELEASE_SCHEMA = "bvb-release/v1"
COHORT = "bvb"

DEFAULT_RELEASE_DIR = REPO_ROOT / "bvb-data" / "BankerVerifierBench"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "datasets" / "gandalf"
DEFAULT_REWARDKIT_OUTPUT_DIR = REPO_ROOT / "datasets" / "rewardkit"

MCP_PROFILE_PATH = Path("tool-profiles/mcp-server")
SHARED_TOOLS_DIR = "shared-tools"
EXPECTED_TOOL_ROOTS = (
    "market_data_platform",
    "company_brand_assets",
    "sec_edgar/data",
)


def resolve_path(path: Path) -> Path:
    """Resolve relative paths against the repository root."""
    return path if path.is_absolute() else REPO_ROOT / path

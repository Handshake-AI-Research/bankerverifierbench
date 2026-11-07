#!/bin/bash
# bootstrap.sh — set up the isolated environments of the host-side verifiers.
#
# Usage: bootstrap.sh [aaaj] [archipelago] [autorubric]   (default: all three)
#
# Idempotent: re-running just verifies the existing setup.
#
# Prerequisites (you set these up yourself):
#   * uv on PATH (https://docs.astral.sh/uv/)
#   * Harbor on PATH (for the `stage` command)
#   * .env file at $REPO_ROOT/.env with API keys (see .env.example)
#
# Each verifier has its own uv project and Python version because their
# LiteLLM and Python constraints conflict. This script installs the Python
# version each selected verifier needs and syncs its venv from its lockfile.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT="$(cd "$HERE/.." && pwd)"
REPO_ROOT="$(cd "$EVAL_ROOT/.." && pwd)"

if [[ $# -eq 0 ]]; then
    set -- aaaj archipelago autorubric
fi

echo "==> verifier-eval bootstrap: $*"
echo "    eval root: $EVAL_ROOT"
echo "    repo root: $REPO_ROOT"
echo

# --- Tool checks --------------------------------------------------------

if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: 'uv' not on PATH; install it first." >&2
    exit 2
fi
echo "    uv:      $(uv --version 2>&1 | head -1)"

if ! command -v harbor >/dev/null 2>&1; then
    echo "    harbor:  NOT FOUND (only needed for the 'stage' command)" >&2
fi

# --- Sync per-verifier venvs -------------------------------------------

for verifier in "$@"; do
    project_dir="$EVAL_ROOT/verifiers/$verifier"
    case "$verifier" in
        aaaj) python_version="3.11" ;;
        archipelago) python_version="3.13"; project_dir="$project_dir/grading" ;;
        autorubric) python_version="3.12" ;;
        *) echo "ERROR: unknown host-side verifier '$verifier' (aaaj, archipelago, autorubric)" >&2; exit 2 ;;
    esac
    if [[ ! -f "$project_dir/pyproject.toml" ]]; then
        echo "ERROR: $project_dir/pyproject.toml missing" >&2
        exit 2
    fi

    echo
    echo "==> uv sync $project_dir (Python $python_version)"
    uv python install "$python_version"
    ( cd "$project_dir" && uv sync --frozen --python "$python_version" --quiet )
    echo "    ok: $verifier venv ready at $project_dir/.venv ($( "$project_dir/.venv/bin/python" --version ))"
done

# --- API key check -----------------------------------------------------

echo
echo "==> .env / API key check"
if [[ ! -f "$REPO_ROOT/.env" ]]; then
    echo "    WARNING: $REPO_ROOT/.env not found; see .env.example." >&2
    echo "    Without it the runners fail at the first model call unless keys are exported." >&2
else
    echo "    ok: $REPO_ROOT/.env exists"
    echo "    (presence of specific keys is logged in each run's manifest.json)"
fi

echo
echo "==> bootstrap complete"

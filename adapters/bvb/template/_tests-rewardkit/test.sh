#!/bin/bash
# RewardKit verifier entrypoint. Harbor runs this as the verifier user.

set -euo pipefail

if [ "${BVB_REWARDKIT_AGENT:-claude-code}" = codex ]; then
  exec /opt/agent-python/bin/python3 /tests/run_codex_judge.py
fi

mkdir -p /logs/verifier /logs/agent/workspace /home/verifier/workspace
cp -a /home/agent/workspace/. /logs/agent/workspace/ \
  2>/logs/agent/workspace-copy-stderr.txt || true
JUDGE_WORKSPACE=/home/verifier/workspace
if [ ! -f /tests/input-contract.json ]; then
  cp -a /home/agent/workspace/. "${JUDGE_WORKSPACE}/" \
    2>/logs/verifier/workspace-copy-stderr.txt
fi

if [ ! -f /logs/agent/trajectory.json ]; then
  cat > /logs/agent/trajectory.json <<'TRAJ'
{"schema_version": "ATIF-v1.4", "session_id": "verify-only", "agent": {"name": "nop", "version": "1.0.0"}, "steps": [{"step_id": 1, "source": "system", "message": "Verify-only: artifacts pre-loaded."}]}
TRAJ
fi

# Validate exact human-review inputs before any paid judge call.
if [ -f /tests/input-contract.json ]; then
  JUDGE_WORKSPACE=/home/verifier/review-inputs
  python3 /tests/bvb_contract.py --prepare-view --replay-view "${JUDGE_WORKSPACE}" || exit 2
  python3 /tests/bvb_contract.py --check-mcp --replay-view "${JUDGE_WORKSPACE}" || exit 2
fi

DELIVERABLES_DIR="/home/agent/workspace/banker_workspace/deliverables"
if [ ! -d "${DELIVERABLES_DIR}" ] \
   || [ -z "$(find "${DELIVERABLES_DIR}" -type f 2>/dev/null)" ]; then
  echo "[verifier] No deliverables found — returning zero without a judge call"
  echo '{"rubric": 0.0}' > /logs/verifier/reward.json
  echo '{"reason": "empty_workspace", "rewards": {}}' > /logs/verifier/reward-details.json
  exit 0
fi

# RewardKit's Claude Code judge must be able to inspect the files and use the
# same BVB MCP source as Gandalf. The wrapper is setuid to the environment user.
mkdir -p "${HOME:-/home/verifier}/.claude"
python3 - <<'PY'
import json
import os
from pathlib import Path

home = Path(os.environ.get("HOME", "/home/verifier"))
mcp_path = home / ".claude.json"
try:
    config = json.loads(mcp_path.read_text()) if mcp_path.exists() else {}
except json.JSONDecodeError:
    config = {}
mcp_config_path = Path("/tests/mcp-config.json")
server = json.loads(mcp_config_path.read_text()) if mcp_config_path.exists() else {"command": "/usr/bin/mcp-server", "args": []}
config.setdefault("mcpServers", {})["mcp-server"] = {
    "type": "stdio", "command": server["command"], "args": server.get("args", []),
}
mcp_path.write_text(json.dumps(config, indent=2))

settings_path = home / ".claude" / "settings.json"
try:
    settings = json.loads(settings_path.read_text()) if settings_path.exists() else {}
except json.JSONDecodeError:
    settings = {}
settings.setdefault("permissions", {})["defaultMode"] = "bypassPermissions"
settings["skipDangerousModePermissionPrompt"] = True
settings_path.write_text(json.dumps(settings, indent=2))
PY

export PATH="/opt/agent-python/bin:${PATH}"

set +e
# Keep the package/version marker here as well as in the image so generated
# tasks and run manifests retain the exact RewardKit contract.
rewardkit /tests \
  --workspace "${JUDGE_WORKSPACE}" \
  --output /logs/verifier/reward.json
# harbor-rewardkit==0.1.2
rewardkit_status=$?
set -e

session_dir="${HOME:-/home/verifier}/.claude/projects/$(echo "${JUDGE_WORKSPACE}" | tr / -)"
trace_dir="/logs/verifier/judge_trajectories"
mkdir -p "${trace_dir}"
if [ -d "${session_dir}" ]; then
  cp -a "${session_dir}/." "${trace_dir}/" \
    2>/logs/verifier/judge-trajectory-copy-stderr.txt || true
fi

exit "${rewardkit_status}"

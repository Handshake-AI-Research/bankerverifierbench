"""Trusted outer verifier: copy reviewed evidence, then run native RewardKit as sandbox."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import tomllib

import tomli_w
from bvb_contract import (REPLAY_VIEW, prepare_replay_view, validate_runtime,
                          tree_hash, file_hash, EXPECTED_MCP_TOOLS)


def main():
    if os.getuid() != 10002:
        raise RuntimeError("RewardKit outer verifier must be UID 10002")
    tests = Path("/tests")
    output = Path("/logs/verifier")
    output.mkdir(exist_ok=True)
    env = dict(os.environ, PATH="/opt/agent-python/bin:/usr/local/bin:/usr/bin:/bin")
    view = Path(REPLAY_VIEW)
    prepare_replay_view(Path("/home/agent/workspace"), Path("/logs/agent/trajectory.json"), view)
    subprocess.run(["sudo", "-n", "-u", "sandbox", "env", "HOME=/tmp",
                    "PATH=" + env["PATH"], "python3", "/tests/bvb_contract.py",
                    "--check-mcp-only", "--output", str(output / "mcp-audit.json")], check=True)
    audit = validate_runtime(tests, Path("/home/agent/workspace"),
                             Path("/logs/agent/trajectory.json"), output / "input-audit.json",
                             replay_view=view, mcp_audit=output / "mcp-audit.json")
    private = Path(tempfile.mkdtemp(prefix="bvb_rewardkit_"))
    private.chmod(0o2770)
    evidence = private / "evidence"
    shutil.copytree(view, evidence, symlinks=False)
    native_tests = private / "tests"
    # RewardKit imports every Python file during discovery. Infrastructure
    # entrypoints belong to the outer verifier, not native reward discovery.
    shutil.copytree(tests, native_tests,
                    ignore=shutil.ignore_patterns("run_codex_judge.py", "bvb_contract.py", "__pycache__"))
    config_path = native_tests / "rubric/criteria.toml"
    config = tomllib.loads(config_path.read_text())
    config["judge"]["atif-trajectory"] = str(evidence / "trajectory.json")
    config_path.write_text(tomli_w.dumps(config))
    if (tree_hash(evidence / "workspace") != audit["identity"]["workspace_sha256"]
            or file_hash(evidence / "trajectory.json") != audit["identity"]["trajectory_sha256"]):
        raise RuntimeError("Private judge copy differs from reviewed evidence")
    # Codex refuses to install its helper aliases below /tmp. Runtime metadata
    # stays outside the evidence copy, in a per-trial verifier-owned directory.
    code_home = Path(tempfile.mkdtemp(prefix="bvb_codex_", dir="/home/verifier"))
    server = json.loads((tests / "mcp-config.json").read_text())
    effort = os.environ.get("BVB_REWARDKIT_REASONING_EFFORT", "high")
    (code_home / "config.toml").write_text(tomli_w.dumps({
        "model_reasoning_effort": effort,
        "approval_policy": "never",
        # Daytona refuses nested user namespaces, and Codex 0.160.1's legacy
        # fallback refuses filesystem-restricted execution. RLE remains the
        # enforcing sandbox: unprivileged UID 10003, no agent group, a private
        # replay copy, and protected source/cache directories. This setting
        # applies only inside that container, after the boundary probe below.
        "sandbox_mode": "danger-full-access",
        "features": {"api_key_model_discovery": True, "multi_agent": False},
        "mcp_servers": {"bvb": {"command": server["command"], "args": server.get("args", []),
                                    "default_tools_approval_mode": "approve",
                                    "enabled_tools": sorted(EXPECTED_MCP_TOOLS),
                                    "startup_timeout_sec": 120}},
    }))
    for directory in (private, code_home):
        for root, dirs, files in os.walk(directory):
            Path(root).chmod(0o2770)
            for name in files:
                (Path(root) / name).chmod(0o660)
    probe_source = '''
import json, os, subprocess
from pathlib import Path
assert os.getuid() == 10003 and 10001 not in os.getgroups()
assert list(Path(os.environ['BVB_JUDGE_COPY']).rglob('*'))
assert subprocess.run(['sudo', '-n', 'id'], capture_output=True).returncode != 0
for path in ['/home/agent/workspace', '/opt/mcp-server', '/var/cache/bvb-tools']:
    try:
        list(Path(path).iterdir())
    except PermissionError:
        continue
    raise RuntimeError('Inner judge can directly access protected original data: ' + path)
print(json.dumps({'uid': os.getuid(), 'groups': os.getgroups(), 'original_agent_workspace_denied': True,
                  'private_mcp_source_and_cache_denied': True, 'judge_copy_accessible': True,
                  'root_escalation_denied': True}))
'''
    boundary = subprocess.run(
        ["sudo", "-n", "-u", "sandbox", "env", "BVB_JUDGE_COPY=" + str(evidence),
         "PATH=" + env["PATH"], "python3", "-c", probe_source],
        check=True, text=True, capture_output=True,
    )
    (output / "isolation-audit.json").write_text(boundary.stdout)
    env.update(CODEX_HOME=str(code_home), HOME="/home/sandbox",
               BVB_REWARDKIT_TRACE_DIR=str(output / "judge_trajectories"))
    Path(env["BVB_REWARDKIT_TRACE_DIR"]).mkdir(exist_ok=True)
    Path(env["BVB_REWARDKIT_TRACE_DIR"]).chmod(0o2770)
    retained = "OPENAI_API_KEY,CODEX_HOME,HOME,PATH,BVB_REWARDKIT_TRACE_DIR,BVB_REWARDKIT_REASONING_EFFORT"
    # API-mode Codex reads its auth store, rather than automatically using an
    # inherited OPENAI_API_KEY. Keep that store outside reviewed evidence and
    # never copy it into exported judge logs.
    login = subprocess.run(
        ["sudo", "-n", "-u", "sandbox", "--preserve-env=" + retained,
         "codex", "login", "--with-api-key"],
        input=env["OPENAI_API_KEY"], text=True, capture_output=True, env=env,
        timeout=60,
    )
    if login.returncode:
        raise RuntimeError("Codex API login failed; credential-bearing auth files are not exported")
    command = ["sudo", "-n", "-u", "sandbox", "--preserve-env=" + retained,
               "rewardkit", str(native_tests), "--workspace", str(evidence),
               "--output", str(output / "reward.json")]
    (output / "judge-config.json").write_text(json.dumps({
        "agent": "codex", "model": config["judge"]["model"], "reasoning_effort": effort,
        "native_rewardkit_isolated": config["judge"]["isolated"],
        "boundary": "Private materialized replay copy; judge UID 10003 without agent group",
        "original_workspace": "/home/agent/workspace", "judge_workspace": str(evidence),
        "judge_trajectory": str(evidence / "trajectory.json"), "criteria": config["criterion"],
        "codex_home": str(code_home),
        "codex_sandbox": "danger-full-access inside the enforcing RLE/Daytona sandbox only",
        "codex_network": "existing container network",
        "sandbox_compatibility": "Codex bubblewrap uid-map failure; legacy Landlock rejected by native CLI",
        "multi_agent": False,
        "api_key_model_discovery": True,
        "mcp_approval": "explicit approval for the configured BVB server only",
        "prompt_template": (native_tests / "judge-prompt.md").read_text(),
    }, indent=2))
    try:
        result = subprocess.run(command, env=env, check=False)
    finally:
        sessions = code_home / "sessions"
        if sessions.is_dir():
            subprocess.run(["sudo", "-n", "-u", "sandbox", "cp", "-a", str(sessions),
                            str(output / "judge_trajectories")], check=True)
            subprocess.run(["sudo", "-n", "-u", "sandbox", "chmod", "-R", "g+rX",
                            str(output / "judge_trajectories/sessions")], check=True)
    validate_runtime(tests, Path("/home/agent/workspace"), Path("/logs/agent/trajectory.json"),
                     output / "input-audit-after.json", replay_view=view,
                     mcp_audit=output / "mcp-audit.json")
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()

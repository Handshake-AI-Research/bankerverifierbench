#!/usr/bin/env python3
"""Read real shared-tool archive members inside generated local Docker images.

Builds each generated task's environment image and calls its configured MCP
server over stdio as the judge (UID 10003). Checks that the judge can read and
hash-verify exports from all three tool families, while the source and cache
remain private. Makes no model calls.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tomllib
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapters.bvb.config import DEFAULT_RELEASE_DIR
from adapters.bvb.contract import EXPECTED_MCP_TOOLS
from adapters.bvb.release import BVBRelease, ReleaseError

PROBE = r"""
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import tempfile
import threading
import time

if os.getuid() != 10003 or 10001 in os.getgroups():
    raise RuntimeError("Probe must run as the isolated judge UID 10003, without the agent group")
for private in ("/opt/mcp-server", "/var/cache/bvb-tools"):
    try:
        list(Path(private).iterdir())
    except PermissionError:
        continue
    raise RuntimeError("Judge can directly read private source or cache: " + private)

server = json.loads(os.environ["BVB_PROBE_SERVER"])
members = json.loads(os.environ["BVB_PROBE_MEMBERS"])
expected_tools = set(json.loads(os.environ["BVB_PROBE_EXPECTED_TOOLS"]))
workspace = Path("/home/verifier/workspace/tool-smoke")
responses = queue.Queue()

with tempfile.TemporaryFile(mode="w+t") as errors:
    process = subprocess.Popen(
        [server["command"], *server.get("args", [])],
        # Match the inner-judge preflight: lazy archive settings must reach
        # the wrapper through the configured command, not inherited globals.
        env={"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/tmp"},
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors,
        text=True, bufsize=1,
    )

    def read_responses():
        for line in process.stdout:
            responses.put(line)
        responses.put(None)

    threading.Thread(target=read_responses, daemon=True).start()

    def send(message):
        process.stdin.write(json.dumps(message) + "\n")
        process.stdin.flush()

    def request(ident, method, params):
        send(dict(jsonrpc="2.0", id=ident, method=method, params=params))
        deadline = time.monotonic() + 180
        while True:
            line = responses.get(timeout=max(0, deadline - time.monotonic()))
            if line is None:
                raise RuntimeError("MCP server exited before responding to " + method)
            response = json.loads(line)
            if response.get("id") != ident:
                continue
            if "error" in response:
                raise RuntimeError("MCP request failed: " + str(response["error"]))
            result = response["result"]
            if result.get("isError"):
                raise RuntimeError("MCP tool failed: " + str(result))
            return result

    try:
        request(1, "initialize", dict(
            protocolVersion="2025-06-18", capabilities={},
            clientInfo=dict(name="bvb-tool-smoke", version="1"),
        ))
        send(dict(jsonrpc="2.0", method="notifications/initialized", params={}))
        tools = request(2, "tools/list", {})
        if {tool["name"] for tool in tools["tools"]} != expected_tools:
            raise RuntimeError("MCP tool list differs from the benchmark contract")
        results = {}
        for ident, (family, member) in enumerate(members.items(), 3):
            result = request(ident, "tools/call", dict(
                name=member["tool"],
                arguments=dict(member["arguments"], workspace_path=str(workspace)),
            ))
            value = next(json.loads(c["text"]) for c in result["content"] if c["type"] == "text")
            if value.get("success") is not True:
                raise RuntimeError("MCP export failed: " + str(value))
            paths = value.get("filepaths", []) or ([value["filepath"]] if value.get("filepath") else [])
            if len(paths) != 1:
                raise RuntimeError("Expected exactly one exported file: " + str(value))
            exported = Path(paths[0]).resolve()
            if not exported.is_relative_to(workspace):
                raise RuntimeError("Tool returned a path outside its export workspace")
            # Read as the judge, not the environment user that owns the cache.
            digest = hashlib.sha256(exported.read_bytes()).hexdigest()
            if digest != member["sha256"]:
                raise RuntimeError("Export hash mismatch: " + str(exported))
            results[family] = dict(
                logical_path=member["logical_path"], archive=member["archive_path"],
                tool=member["tool"], filepath=str(exported), sha256=digest,
                mode=format(exported.stat().st_mode & 0o777, "04o"), ok=True,
            )
        print(json.dumps(dict(caller_uid=os.getuid(), caller_groups=os.getgroups(),
                              private_source_and_cache_denied=True,
                              mcp_tools=sorted(expected_tools), families=results)))
    except Exception as error:
        errors.flush()
        errors.seek(0)
        raise RuntimeError(str(error) + "\nMCP stderr:\n" + errors.read()[-4000:]) from error
    finally:
        # EOF lets the privileged wrapper's server exit without granting the
        # judge permission to signal processes owned by the environment user.
        process.stdin.close()
        process.wait(timeout=15)
"""


def probe_members(index: dict) -> dict[str, dict]:
    """Select one real tool export per family, covering gzip and zstd archives."""
    members = index["members"]
    logo = index["api_indexes"]["company_brand_assets"]["logos"][0]
    families = {
        "company_brand_assets": (logo["logical_path"], "get_logo_file", {"identifier": logo["stem"]}),
        "market_data_platform": ("market_data_platform/_wacc/betas_for_wacc.xlsx", "get_industry_betas", {}),
        "sec_edgar": (index["api_indexes"]["sec_edgar"]["cik_lookup"], "copy_cik_lookup", {}),
    }
    selected = {family: dict(logical_path=path, tool=tool, arguments=arguments, **members[path])
                for family, (path, tool, arguments) in families.items()}
    formats = {archive["format"] for archive in index["archives"]
               if archive["path"] in {item["archive_path"] for item in selected.values()}}
    if formats != {"tar.gz", "tar.zst"}:
        raise ReleaseError("Probe members must cover both gzip and zstd archives")
    return selected


def check_image(task_dir: Path, members: dict) -> dict:
    environment = task_dir / "environment"
    compose_path = environment / "docker-compose.yaml"
    if not compose_path.is_file():
        raise ReleaseError(f"{task_dir.name}: no local docker-compose.yaml; generate the task for Docker")
    service = yaml.safe_load(compose_path.read_text())["services"]["main"]
    server = tomllib.loads((task_dir / "task.toml").read_text())["environment"]["mcp_servers"][0]
    tag = f"bvb-tool-smoke:{task_dir.parent.parent.name}-{task_dir.name}".lower()
    subprocess.run(["docker", "build", "--quiet", "-t", tag, str(environment)], check=True,
                   stdout=subprocess.DEVNULL)
    command = ["docker", "run", "--rm", "--network", "none", "--user", "sandbox",
               "-e", "HOME=/tmp", "-e", f"BVB_PROBE_MEMBERS={json.dumps(members)}",
               "-e", f"BVB_PROBE_SERVER={json.dumps(server)}",
               "-e", f"BVB_PROBE_EXPECTED_TOOLS={json.dumps(sorted(EXPECTED_MCP_TOOLS))}"]
    for volume in service["volumes"]:
        command += ["-v", volume]
    for key, value in service["environment"].items():
        command += ["-e", f"{key}={value}"]
    command += [tag, "/opt/agent-python/bin/python", "-c", PROBE]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise ReleaseError(f"{task_dir}: judge MCP probe failed:\n{result.stderr.strip()}")
    observed = json.loads(result.stdout.strip().splitlines()[-1])
    if (observed.get("caller_uid") != 10003 or not observed.get("private_source_and_cache_denied")
            or set(observed.get("families", {})) != set(members)
            or not all(item["ok"] for item in observed["families"].values())):
        raise ReleaseError(f"{task_dir}: incomplete judge MCP probe: {observed}")
    return {"task_dir": str(task_dir), "image": tag, "status": "passed", **observed}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE_DIR)
    parser.add_argument("--task-dir", type=Path, action="append", required=True,
                        help="Generated task directory; repeat for several images.")
    args = parser.parse_args()
    release = BVBRelease.open(args.release_dir)
    members = probe_members(release.tool_archive_index(require_archive_files=True))
    for task_dir in args.task_dir:
        print(json.dumps(check_image(task_dir.resolve(), members), sort_keys=True))


if __name__ == "__main__":
    main()

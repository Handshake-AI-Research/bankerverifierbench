#!/opt/agent-python/bin/python3
"""Record native Codex stderr without changing its structured stdout contract."""
import os
from pathlib import Path
import subprocess
import sys
import uuid

args = sys.argv[1:]
if args and args[0] == "exec":
    args[1:1] = ["--skip-git-repo-check"]
directory = os.environ.get("BVB_REWARDKIT_TRACE_DIR")
if not directory or not args or args[0] != "exec":
    os.execv("/usr/local/bin/codex-native", ["codex", *args])
path = Path(directory) / ("codex-" + uuid.uuid4().hex + "-stderr.txt")
with path.open("wb") as log:
    # RewardKit supplies the complete prompt as argv. A non-TTY inherited
    # Harbor stdin can remain open forever; Codex otherwise waits for EOF.
    process = subprocess.Popen(["/usr/local/bin/codex-native", *args],
                               stdin=subprocess.DEVNULL, stderr=subprocess.PIPE)
    for line in iter(process.stderr.readline, b""):
        log.write(line)
        log.flush()
        sys.stderr.buffer.write(line)
        sys.stderr.buffer.flush()
    raise SystemExit(process.wait())

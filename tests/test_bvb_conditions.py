import csv
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from adapters.bvb.config import DEFAULT_RELEASE_DIR
from adapters.bvb import runtime_snapshots
from adapters.bvb.daytona import snapshot_revision_label
from adapters.bvb.cli import command, load_conditions
from scripts.run_bvb_verifier_matrix import run_one

ROOT = Path(__file__).resolve().parents[1]
RELEASE = Path(os.environ.get("BVB_TEST_RELEASE", DEFAULT_RELEASE_DIR))
OUT = ROOT / "outputs" / "run"
BASE_SNAPSHOT = "bvb-re8068910-m07ab719f-iad90681e-aa3364324-t8d4c445a"


class ConditionRegistryTests(unittest.TestCase):
    def setUp(self):
        self.conditions = load_conditions()

    def test_registry_covers_exactly_the_bundle_conditions(self):
        with (ROOT / "results/paper/conditions.csv").open() as stream:
            rows = {row["id"]: row for row in csv.DictReader(stream)}
        self.assertEqual(set(self.conditions), set(rows))
        for cid, condition in self.conditions.items():
            row = rows[cid]
            self.assertEqual(condition["verifier"], row["verifier"], cid)
            self.assertEqual(condition["model"].split("/", 1)[1], row["model"], cid)
            self.assertEqual(condition["repeats"] * 24, int(row["target"]), cid)

    def test_ported_conditions_build_their_runner_command(self):
        for cid, condition in self.conditions.items():
            if condition["runner"] in {"gandalf-daytona", "gandalf-batching"}:
                continue
            cmd = command(condition, output=OUT, tasks=["btb-x"], repeats=1,
                          evidence_run=Path("/out/m01"))
            self.assertTrue((ROOT / cmd[1]).is_file(), cid)
            self.assertEqual(cmd[cmd.index("--model") + 1], condition["model"], cid)
            self.assertEqual(cmd[cmd.index("--attempts") + 1], "1", cid)
            if condition["runner"] == "host-matrix":
                self.assertEqual(cmd[cmd.index("--verifiers") + 1], condition["verifier"], cid)
                self.assertEqual(cmd[cmd.index("--dataset") + 1], "/out/m01/bvb", cid)
                self.assertEqual("--litellm-version" in cmd, "litellm_version" in condition, cid)

    def test_commands_use_the_requested_release(self):
        custom = Path("/data/bvb-release")
        for cid, condition in self.conditions.items():
            if condition["runner"] == "gandalf-daytona":
                continue
            cmd = command(condition, output=OUT, evidence_run=Path("/out/m01"), release_dir=custom)
            flag = "--release" if condition["runner"] == "host-matrix" else "--release-dir"
            self.assertEqual(cmd[cmd.index(flag) + 1], str(custom), cid)

    def test_staging_runners_require_an_output_inside_the_repository(self):
        with self.assertRaisesRegex(ValueError, "inside the repository"):
            command(self.conditions["M05"], output=Path("/tmp/m05"))

    def test_host_conditions_require_staged_evidence(self):
        with self.assertRaisesRegex(ValueError, "evidence-run"):
            command(self.conditions["M13"], output=OUT)

    def test_batching_conditions_pass_their_partition(self):
        cmd = command(self.conditions["B-b16"], output=OUT, tasks=["btb-x"], repeats=1)
        self.assertEqual(cmd[1], "scripts/run_bvb_gandalf_batching.py")
        self.assertEqual(cmd[cmd.index("--batch-size") + 1], "16")
        self.assertEqual(cmd[cmd.index("--attempts") + 1], "1")
        cmd = command(self.conditions["B-split2"], output=OUT)
        self.assertEqual(cmd[cmd.index("--batch-splits") + 1], "2")
        self.assertNotIn("--batch-size", cmd)

    def test_batching_runner_rejects_other_models(self):
        condition = dict(self.conditions["B-b16"], model="openai/gpt-5.4")
        with self.assertRaisesRegex(ValueError, "batching runner"):
            command(condition, output=OUT)

    def test_repeats_stay_within_benchmark_slots(self):
        with self.assertRaises(ValueError):
            command(self.conditions["M05"], output=OUT, repeats=4)


class RuntimeSnapshotTests(unittest.TestCase):
    def test_plans_derive_stable_runtime_names(self):
        self.assertEqual(
            runtime_snapshots.plan("gandalf-llm", base_snapshot=BASE_SNAPSHOT)["snapshot"],
            BASE_SNAPSHOT + "-llm-13ce00af",
        )
        self.assertEqual(
            runtime_snapshots.plan("rewardkit", base_snapshot=BASE_SNAPSHOT)["snapshot"],
            BASE_SNAPSHOT + "-rewardkit-a64a78c8",
        )

    @unittest.skipUnless((RELEASE / "release.json").exists(), "BVB release not fetched")
    def test_logo_patch_reproduces_the_measured_mcp_source(self):
        from adapters.bvb.release import BVBRelease

        release = BVBRelease.open(RELEASE)
        original = (release.mcp_root / "tools" / "company_brand_assets.py").read_bytes()
        runtime_snapshots.patched_bytes(
            original, runtime_snapshots.LOGO_PATCH, runtime_snapshots.LOGO_EXPORT_PERMISSIONS
        )
        plan = runtime_snapshots.plan("gandalf-llm", release)
        self.assertTrue(plan["snapshot"].startswith(f"bvb-r{snapshot_revision_label(release.revision)}-"))

    def test_patch_rejects_unexpected_original(self):
        with self.assertRaises(Exception):
            runtime_snapshots.patched_bytes(
                b"other", runtime_snapshots.LOGO_PATCH, runtime_snapshots.LOGO_EXPORT_PERMISSIONS
            )


class HostRunnerTests(unittest.TestCase):
    def test_package_cache_is_pinned_before_per_run_cache_and_preserves_overrides(self):
        from verifier_runner.runners._runtime import pin_dependency_cache

        env = {}
        with patch("verifier_runner.runners._runtime.subprocess.check_output",
                   return_value="/tmp/shared-uv-cache\n") as probe:
            pin_dependency_cache(env, "1.104.0")
            self.assertEqual(env["UV_CACHE_DIR"], "/tmp/shared-uv-cache")
            env["XDG_CACHE_HOME"] = "/tmp/new-run-artifacts"
            pin_dependency_cache(env, "1.104.0")
            probe.assert_called_once()
            self.assertEqual(env["UV_CACHE_DIR"], "/tmp/shared-uv-cache")
        baseline = {}
        with patch("verifier_runner.runners._runtime.subprocess.check_output") as probe:
            pin_dependency_cache(baseline, None)
            probe.assert_not_called()
            self.assertEqual(baseline, {})

    def test_stop_control_prevents_new_attempt_without_entering_grader(self):
        with tempfile.TemporaryDirectory() as tmp:
            stop = Path(tmp) / "stop"
            stop.touch()
            args = SimpleNamespace(stop_file=stop, model="openai/gpt-5.4")
            with patch("scripts.run_bvb_verifier_matrix.resolve") as resolve:
                result = run_one(args, "autorubric", "task", 0)
                resolve.assert_not_called()
            self.assertEqual(result["status"], "not_started_control_stop")
            self.assertEqual(result["repeat"], 1)


if __name__ == "__main__":
    unittest.main()

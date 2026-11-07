"""Tests for the fully parallel BVB Daytona execution path."""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "verifier-eval")]

from adapters.bvb.adapter import TEMPLATE_DIR, runtime_template_sha256
from adapters.bvb.daytona import _snapshot_identity, snapshot_name, snapshot_revision_label
from adapters.bvb.release import ReleaseError
from scripts.run_bvb_gandalf_daytona import _validate_judge_trace, main, validate_job
from verifier_runner.stage import stage


class DaytonaFullRunTests(unittest.TestCase):
    def test_trace_validation_accepts_reasoning_without_effort_level(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            trace = Path(temporary) / "trace.txt"
            trace.write_text(
                "Tools Available: 19\n"
                "Created 15 MCP tools\n"
                "'model': 'openrouter/qwen/qwen3.8-flash'\n"
                "'reasoning_effort': None\n"
                "'extended_thinking_budget': 131072\n"
                "'litellm_extra_body': {'reasoning': {'enabled': True, "
                "'max_tokens': 131072}}\n"
            )
            _validate_judge_trace(
                trace,
                verifier_model="openrouter/qwen/qwen3.8-flash",
                reasoning_config={
                    "reasoning_effort": None,
                    "extended_thinking_budget": 131072,
                    "reasoning_max_tokens": 131072,
                },
            )

    def test_trace_validation_accepts_rich_wrapped_model_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            trace = Path(temporary) / "trace.txt"
            trace.write_text(
                "Tools Available: 19\n"
                "Created 15 MCP tools\n"
                "'model': 'openrouter/deepseek/deepseek-v4-flash\n"
                "          -0731'\n"
                "'reasoning_effort': 'high'\n"
                "'extended_thinking_budget': 200000\n"
                "'litellm_extra_body': {'provider': {'only': ['coreweave'], "
                "'allow_fallbacks': False, 'require_parameters': True}}\n"
            )
            _validate_judge_trace(
                trace,
                verifier_model="openrouter/deepseek/deepseek-v4-flash-0731",
                reasoning_config={
                    "reasoning_effort": "high",
                    "extended_thinking_budget": 200000,
                },
                provider_routing={
                    "only": ["coreweave"],
                    "allow_fallbacks": False,
                    "require_parameters": True,
                },
            )

    def test_cli_model_reaches_adapter_and_run_summary(self) -> None:
        model = "gemini/gemini-3.6-flash"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            jobs_dir = root / "jobs"
            jobs_dir.mkdir()
            fake_release = SimpleNamespace(
                root=root / "release",
                select_tasks=lambda cohort, task_ids: [
                    SimpleNamespace(task_id="btb-one")
                ],
            )
            adapter = MagicMock()
            adapter.generate.return_value = [
                SimpleNamespace(
                    task_id="btb-one",
                    trajectory_provenance="authentic_agent_trajectory",
                )
            ]
            stage_result = SimpleNamespace(returncode=0, jobs_dir=jobs_dir)
            output = io.StringIO()
            with (
                patch(
                    "scripts.run_bvb_gandalf_daytona.validate_release",
                    return_value=(fake_release, None),
                ),
                patch(
                    "scripts.run_bvb_gandalf_daytona.validate_snapshot_runtime",
                    return_value={"ok": True},
                ),
                patch(
                    "scripts.run_bvb_gandalf_daytona.BVBAdapter",
                    return_value=adapter,
                ) as adapter_class,
                patch(
                    "scripts.run_bvb_gandalf_daytona.stage",
                    return_value=stage_result,
                ),
                patch(
                    "scripts.run_bvb_gandalf_daytona.validate_job",
                    return_value={"trial_results": []},
                ) as validate,
                contextlib.redirect_stdout(output),
            ):
                return_code = main(
                    [
                        "--release-dir",
                        str(root / "release"),
                        "--output-dir",
                        str(root / "dataset"),
                        "--snapshot",
                        "test-snapshot",
                        "--run-id",
                        "unit-model-propagation",
                        "--task-ids",
                        "btb-one",
                        "--attempts",
                        "1",
                        "--concurrency",
                        "1",
                        "--verifier-model",
                        model,
                    ]
                )
            self.assertEqual(return_code, 0)
            self.assertEqual(adapter_class.call_args.kwargs["verifier_model"], model)
            self.assertEqual(validate.call_args.kwargs["verifier_model"], model)
            summary = json.loads((jobs_dir / "bvb-run-summary.json").read_text())
            self.assertEqual(summary["verifier_model"], model)
            self.assertEqual(
                summary["effective_reasoning_configuration"]["reasoning_effort"],
                "high",
            )

    def test_snapshot_identity_tracks_runtime_and_inner_judge_configuration(self) -> None:
        class FakeRelease:
            tool_profile = {"source_tree_sha256": "1" * 64}

            @staticmethod
            def tool_archive_index(require_archive_files: bool = False) -> dict:
                return {
                    "index_sha256": "2" * 64,
                    "archive_set_sha256": "3" * 64,
                }

        release = FakeRelease()
        identity = _snapshot_identity(
            release,
            release_revision="revision",
            cache_max_bytes=123,
        )
        self.assertEqual(identity["schema_version"], "bvb-daytona-snapshot/v2")
        self.assertEqual(identity["runtime_sha256"], runtime_template_sha256())
        self.assertTrue(
            snapshot_name(release, release_revision="revision").endswith(
                f"-t{runtime_template_sha256()[:8]}"
            )
        )
        self.assertTrue(snapshot_name(release, release_revision="v1.0").startswith("bvb-rv1-0-m"))
        self.assertEqual(snapshot_revision_label("0123abcd" + "e" * 32), "0123abcd")

        dockerfile = (TEMPLATE_DIR / "environment" / "Dockerfile").read_text()
        self.assertIn("/opt/agent-python/bin", dockerfile)
        self.assertIn("BVB_TOOL_ARCHIVE_INDEX", dockerfile)
        verifier = (TEMPLATE_DIR / "tests" / "test.sh").read_text()
        self.assertLess(
            verifier.index("Inner-judge Python inspection preflight"),
            verifier.index('export PYTHONPATH="/tests'),
        )
        self.assertLess(
            verifier.index('export PYTHONPATH="/tests'),
            verifier.index("gandalf-the-grader --config"),
        )
        self.assertIn('reward_path.write_text(json.dumps({"reward":', verifier)

    def test_stage_dry_run_forwards_attempts_concurrency_and_retries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            for task_id in ("btb-one", "btb-two"):
                (dataset / task_id).mkdir(parents=True)
            output = io.StringIO()
            with contextlib.redirect_stdout(output), patch(
                "verifier_runner.stage.shutil.which", return_value="/usr/bin/harbor"
            ):
                result = stage(
                    cohort="bvb",
                    tasks=["btb-one", "btb-two"],
                    staging_id="parallel-test",
                    repo_root=root,
                    dataset_dir=dataset,
                    environment_type="daytona",
                    n_concurrent=6,
                    n_attempts=3,
                    max_retries=2,
                    retry_include=["DaytonaError"],
                    environment_kwargs=["snapshot_template_name=test-snapshot"],
                    dry_run=True,
                )
            command = output.getvalue()
            self.assertEqual(result.returncode, 0)
            self.assertIn("-n 6", command)
            self.assertIn("-k 3", command)
            self.assertIn("--max-retries 2", command)
            self.assertIn("--retry-include DaytonaError", command)

    def test_validate_job_requires_three_complete_attempts_per_task(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            jobs = Path(temporary)
            task_ids = ["btb-one", "btb-two"]
            (jobs / "result.json").write_text(
                json.dumps(
                    {
                        "n_total_trials": 6,
                        "stats": {"n_errored_trials": 0},
                    }
                )
            )
            for task_id in task_ids:
                for attempt in range(3):
                    trial = jobs / f"{task_id}__attempt{attempt}"
                    (trial / "agent").mkdir(parents=True)
                    (trial / "verifier").mkdir()
                    (trial / "result.json").write_text(
                        json.dumps(
                            {
                                "task_name": task_id,
                                "exception_info": None,
                                "verifier_result": {"rewards": {"reward": 0.5}},
                            }
                        )
                    )
                    (trial / "agent" / "trajectory.json").write_text("{}")
                    (trial / "verifier" / "info.json").write_text(
                        json.dumps(
                            {
                                "criterion_results": [
                                    {
                                        "criterion": "test",
                                        "weight": 1,
                                        "met": True,
                                    }
                                ],
                                "errored_criterion_count": 0,
                                "evaluated_criteria_pct": 100,
                            }
                        )
                    )
                    (trial / "verifier" / "reward.json").write_text("{}")
                    (trial / "verifier" / "judge_trace_batch_split0.txt").write_text(
                        "Tools Available: 19\nCreated 15 MCP tools\n"
                    )
            summary = validate_job(jobs, task_ids=task_ids, attempts=3)
            self.assertEqual(summary["trials"], 6)
            self.assertTrue(summary["artifacts_complete"])

            trace = (
                jobs
                / "btb-one__attempt0"
                / "verifier"
                / "judge_trace_batch_split0.txt"
            )
            trace.write_text(
                "Tools Available: 4\nCreated 0 MCP tools\n"
                "ModuleNotFoundError: No module named 'openpyxl'\n"
            )
            with self.assertRaisesRegex(ReleaseError, "broken inner-judge runtime"):
                validate_job(jobs, task_ids=task_ids, attempts=3)
            trace.write_text("Tools Available: 19\nCreated 15 MCP tools\n")

            (jobs / "btb-one__attempt0" / "verifier" / "reward.json").unlink()
            with self.assertRaises(ReleaseError):
                validate_job(jobs, task_ids=task_ids, attempts=3)


if __name__ == "__main__":
    unittest.main()

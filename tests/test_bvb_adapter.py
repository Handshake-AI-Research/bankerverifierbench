"""Regression coverage for BVB task materialization, staging lookup and the fail-closed scorer."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "verifier-eval")]

from adapters.bvb.adapter import BVBAdapter, RewardKitBVBAdapter, runtime_template_sha256
from adapters.bvb.config import COHORT, DEFAULT_RELEASE_DIR
from adapters.bvb.contract import input_identity
from adapters.bvb.daytona import snapshot_revision_label
from adapters.bvb.release import BVBRelease, ReleaseError
from verifier_runner.evaluate import DEFAULT_MODEL, _check_skip, _model_for_verifier
from verifier_runner.scoring import load_ground_truth, score_verifier
from verifier_runner.inputs import (
    cohort_dataset_subdir,
    list_staged_trials,
    resolve,
    staging_jobs_subdir,
)
from verifier_runner.runners.base import Capabilities
from verifier_runner.verdict import CriterionVerdict, Verdict


RELEASE_DIR = Path(os.environ.get("BVB_TEST_RELEASE", DEFAULT_RELEASE_DIR))


@unittest.skipUnless((RELEASE_DIR / "shared-tools").is_dir(), "Run `make fetch` to download the release with shared tools")
class BVBAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.release = BVBRelease.open(RELEASE_DIR)
        cls.task_id = cls.release.task_ids(COHORT)[0]

    def test_release_contract_counts(self) -> None:
        result = self.release.validate_hashes(require_all_files=True)
        self.assertEqual(result.task_count, 24)
        self.assertEqual(result.adjudicated_task_count, 24)
        self.assertEqual(result.eligible_criteria_count, 2464)
        self.assertEqual(result.missing_files, ())

    def test_generated_task_uses_canonical_rubric_and_authentic_replay(self) -> None:
        verifier_model = "gemini/gemini-3.5-flash-lite"
        with tempfile.TemporaryDirectory() as temporary:
            generated = BVBAdapter(
                self.release,
                output_dir=Path(temporary) / "generated",
                tool_root=self.release.root,
                verifier_model=verifier_model,
                reasoning_effort="low",
                reasoning_max_tokens=131_072,
                openrouter_provider="coreweave",
            ).generate(self.release.select_tasks(COHORT, [self.task_id]))[0]
            metadata = json.loads(generated.generated_manifest.read_text())
            self.assertEqual(metadata["canonical_rubric_count"], len(self.release.tasks[self.task_id].criteria))
            self.assertEqual(metadata["trajectory_provenance"], "authentic_agent_trajectory")
            self.assertEqual(metadata["workspace_provenance"], "authentic_agent_workspace")
            self.assertEqual(metadata["verifier_model"], verifier_model)
            self.assertEqual(
                metadata["provider_routing"],
                {
                    "only": ["coreweave"],
                    "allow_fallbacks": False,
                    "require_parameters": True,
                },
            )
            self.assertEqual(metadata["effective_reasoning_configuration"]["reasoning_effort"], "low")
            self.assertEqual(metadata["effective_reasoning_configuration"]["reasoning_max_tokens"], 131_072)
            grader = tomllib.loads((generated.task_dir / "tests" / "grader.toml").read_text())
            self.assertEqual(grader["model"], verifier_model)
            sitecustomize = (generated.task_dir / "tests" / "sitecustomize.py").read_text()
            self.assertIn("kwargs['reasoning_effort'] = 'low'", sitecustomize)
            self.assertIn("extra_body.update", sitecustomize)
            self.assertIn("'coreweave'", sitecustomize)
            self.assertIn("'max_tokens': 131072", sitecustomize)
            compile(sitecustomize, "sitecustomize.py", "exec")
            self.assertTrue((generated.task_dir / "solution" / "trajectory.json").is_file())

    def test_rewardkit_task_uses_separate_in_environment_verifier_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generated = RewardKitBVBAdapter(
                self.release,
                output_dir=root / "generated",
                tool_root=self.release.root,
            ).generate(self.release.select_tasks(COHORT, [self.task_id]))[0]

            task_config = tomllib.loads((generated.task_dir / "task.toml").read_text())
            criteria = tomllib.loads((generated.task_dir / "tests" / "rubric" / "criteria.toml").read_text())
            manifest = json.loads(generated.generated_manifest.read_text())

            self.assertEqual(task_config["metadata"]["verifier_kind"], "rewardkit")
            self.assertIn("ANTHROPIC_API_KEY", task_config["verifier"]["env"])
            self.assertEqual(len(criteria["criterion"]), len(self.release.tasks[self.task_id].criteria))
            self.assertFalse((generated.task_dir / "tests" / "grader.toml").exists())
            self.assertEqual(manifest["rewardkit_version"], "0.1.2")
            self.assertIn(
                "@anthropic-ai/claude-code@2.1.246",
                (generated.task_dir / "environment" / "Dockerfile").read_text(),
            )
            with self.assertRaisesRegex(ValueError, "materialized for 'rewardkit'"):
                resolve(
                    COHORT,
                    self.task_id,
                    datasets_root=root,
                    release_root=self.release.root,
                    verifier_kind="gandalf",
                    dataset_dir=generated.task_dir.parent,
                )

    def test_local_logo_correction_preserves_canonical_sources_and_daytona_base(self) -> None:
        from adapters.bvb.daytona import snapshot_name
        from adapters.bvb.release import tree_digest
        from adapters.bvb.runtime_snapshots import LOGO_EXPORT_PERMISSIONS

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for kind, adapter_class in (("gandalf", BVBAdapter), ("rewardkit", RewardKitBVBAdapter)):
                with self.subTest(kind=kind):
                    generated = adapter_class(self.release, output_dir=root / kind,
                                              tool_root=self.release.root).generate(
                        self.release.select_tasks(COHORT, [self.task_id]))[0]
                    metadata = json.loads(generated.generated_manifest.read_text())
                    self.assertEqual(metadata["local_runtime_fixes"],
                                     {"logo_export_permissions": LOGO_EXPORT_PERMISSIONS})
                    self.assertEqual(tree_digest(generated.task_dir / "environment/mcp-server"),
                                     tree_digest(self.release.mcp_root))
                    config = tomllib.loads((generated.task_dir / "task.toml").read_text())
                    self.assertEqual(config["metadata"]["bvb_local_runtime_fixes"], ["logo_export_permissions"])

            generated = BVBAdapter(
                self.release, output_dir=root / "daytona", tool_root=self.release.root,
                execution_environment="daytona", daytona_snapshot=snapshot_name(self.release),
            ).generate(self.release.select_tasks(COHORT, [self.task_id]))[0]
            metadata = json.loads(generated.generated_manifest.read_text())
            self.assertEqual(metadata["local_runtime_fixes"], {})
            from adapters.bvb.adapter import TEMPLATE_DIR
            self.assertEqual((generated.task_dir / "environment/Dockerfile").read_bytes(),
                             (TEMPLATE_DIR / "environment/Dockerfile").read_bytes())

    def test_verifier_kinds_have_distinct_dataset_and_staging_paths(self) -> None:
        self.assertEqual(cohort_dataset_subdir(COHORT, "gandalf"), "gandalf/bvb")
        self.assertEqual(cohort_dataset_subdir(COHORT, "rewardkit"), "rewardkit/bvb")
        self.assertEqual(staging_jobs_subdir(COHORT, "gandalf"), "staging__bvb")
        self.assertEqual(staging_jobs_subdir(COHORT, "rewardkit"), "staging__bvb__rewardkit")

    def test_in_environment_verifiers_keep_their_staged_model(self) -> None:
        self.assertIsNone(_model_for_verifier("gandalf", {}))
        self.assertIsNone(_model_for_verifier("rewardkit", {}))
        self.assertEqual(_model_for_verifier("aaaj", {}), DEFAULT_MODEL)

    def test_repeated_staging_selects_deterministic_trial_index(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generated = BVBAdapter(
                self.release,
                output_dir=root / "generated",
                tool_root=self.release.root,
            ).generate(self.release.select_tasks(COHORT, [self.task_id]))[0]
            staging = root / "staging"
            for suffix in ("z-last", "a-first", "m-middle"):
                agent = staging / f"{self.task_id}__{suffix}" / "agent"
                agent.mkdir(parents=True)
                (agent / "workspace").symlink_to(generated.task_dir / "solution" / "workspace", target_is_directory=True)
                (agent / "trajectory.json").write_bytes(
                    (generated.task_dir / "solution" / "trajectory.json").read_bytes()
                )

            trials = list_staged_trials(staging, self.task_id)
            self.assertEqual(
                [trial.name for trial in trials],
                [f"{self.task_id}__a-first", f"{self.task_id}__m-middle", f"{self.task_id}__z-last"],
            )
            bundle = resolve(
                COHORT,
                self.task_id,
                datasets_root=root,
                release_root=self.release.root,
                staging_jobs_dir=staging,
                trial_index=1,
                dataset_dir=generated.task_dir.parent,
            )
            self.assertEqual(bundle.trial_dir, staging / f"{self.task_id}__m-middle")
            caps = Capabilities(
                needs_workspace=True,
                needs_trajectory=True,
                accepts_placeholder_trajectory=False,
            )
            self.assertIsNone(_check_skip(caps, bundle, "archipelago"))

    def test_unknown_cohort_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown BVB scoring cohort"):
            score_verifier([self._perfect_verdict(self.task_id)], RELEASE_DIR, cohort="all-30")
        with self.assertRaises(ReleaseError):
            self.release.task_ids("unreleased-cohort")

    def test_unhashed_external_agent_artifacts_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            agent = root / "artifacts" / "agent"
            (agent / "workspace").mkdir(parents=True)
            (agent / "workspace" / "scratch.txt").write_text("unverified")
            (agent / "trajectory.json").write_text("{}")
            with patch("adapters.bvb.adapter._find_authentic_agent_dir", return_value=agent):
                with self.assertRaisesRegex(ReleaseError, "matching authentic selected workspace"):
                    BVBAdapter(
                        self.release,
                        output_dir=root / "generated",
                        tool_root=self.release.root,
                        artifact_root=root / "artifacts",
                    ).generate(self.release.select_tasks(COHORT, [self.task_id]))

    def test_daytona_task_uses_digest_named_snapshot_not_local_volume(self) -> None:
        archive_index = {
            "index_sha256": "1" * 64,
            "archive_set_sha256": "2" * 64,
        }
        mcp_digest = self.release.tool_profile["source_tree_sha256"]
        snapshot = (
            f"bvb-r{snapshot_revision_label(self.release.revision)}-m{mcp_digest[:8]}-i{'1' * 8}-a{'2' * 8}"
            f"-t{runtime_template_sha256()[:8]}"
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(self.release, "tool_archive_index", return_value=archive_index):
                result = BVBAdapter(
                    self.release,
                    output_dir=root / "generated",
                    tool_root=self.release.root,
                    execution_environment="daytona",
                    daytona_snapshot=snapshot,
                ).generate(self.release.select_tasks(COHORT, [self.task_id]))[0]
            metadata = json.loads(result.generated_manifest.read_text())
            self.assertEqual(metadata["execution_environment"], "daytona")
            self.assertEqual(metadata["daytona_snapshot"], snapshot)
            self.assertEqual(metadata["daytona_snapshot_identity"]["archive_index_sha256"], "1" * 64)
            self.assertEqual(metadata["daytona_snapshot_identity"]["runtime_sha256"], runtime_template_sha256())
            grader = tomllib.loads((result.task_dir / "tests" / "grader.toml").read_text())
            mcp_server = grader["mcp_servers"][0]
            self.assertEqual(mcp_server["command"], "/usr/bin/env")
            self.assertIn("BVB_TOOL_ARCHIVE_INDEX=/opt/mcp-server/data/tool-archive-index.json", mcp_server["args"])
            self.assertEqual(mcp_server["args"][-1], "/usr/bin/mcp-server")
            self.assertFalse((result.task_dir / "environment" / "docker-compose.yaml").exists())

    def test_daytona_snapshot_without_revision_token_is_rejected(self) -> None:
        stale = "bvb-r00000000-m07ab719f-iad90681e-aa3364324-t8d4c445a"
        with tempfile.TemporaryDirectory() as temporary, self.assertRaisesRegex(ReleaseError, "snapshot name"):
            BVBAdapter(
                self.release,
                output_dir=Path(temporary) / "generated",
                tool_root=self.release.root,
                execution_environment="daytona",
                daytona_snapshot=stale,
            )

    def test_missing_criterion_fails_closed(self) -> None:
        verdicts = [self._perfect_verdict(task_id) for task_id in self.release.task_ids(COHORT)]
        verdicts[0] = verdicts[0].model_copy(update={"criteria": verdicts[0].criteria[:-1]})
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            score_verifier(verdicts, RELEASE_DIR, cohort=COHORT)

    def _perfect_verdict(self, task_id: str) -> Verdict:
        task = self.release.tasks[task_id]
        return Verdict(
            verifier="gandalf",
            model="unit-test",
            task_id=task_id,
            gt_task_id=task_id,
            cohort=COHORT,
            canonical_rubric_sha256=task.manifest["evaluation_rubric"]["sha256"],
            input_identity=input_identity(self.release, task),
            criteria=[
                CriterionVerdict(
                    index=label["evaluation_criterion_index"],
                    evaluation_criterion_index=label["evaluation_criterion_index"],
                    source_criterion_index=label["source_criterion_index"],
                    criterion_text_sha256=label["criterion_text_sha256"],
                    satisfied=label["decision"] == "MET",
                )
                for label in load_ground_truth(self.release, task_id)
            ],
        )


if __name__ == "__main__":
    unittest.main()

import os
import tempfile
import tomllib
import unittest
import urllib.error
from pathlib import Path

from adapters.bvb.adapter import BVBAdapter
from adapters.bvb.batching import (
    MODEL, REASONING_EFFORT, RUNTIME, VERIFIER_TIMEOUT_SEC, BatchingCondition, build_overlay, configure_task, native_source,
)
from adapters.bvb.config import DEFAULT_RELEASE_DIR
from adapters.bvb.release import BVBRelease
from adapters.bvb.runtime_snapshots import plan

RELEASE_DIR = Path(os.environ.get("BVB_TEST_RELEASE", DEFAULT_RELEASE_DIR))


class PartitionTests(unittest.TestCase):
    def test_exact_size_batches_keep_rubric_order(self):
        criteria = list(range(67))
        chunks = BatchingCondition("b16", batch_size=16).partition(criteria)
        self.assertEqual([len(c) for c in chunks], [16, 16, 16, 16, 3])
        self.assertEqual(sum(chunks, []), criteria)

    def test_native_split_halves_the_rubric(self):
        chunks = BatchingCondition("split2", batch_splits=2).partition(list(range(67)))
        self.assertEqual([len(c) for c in chunks], [34, 33])

    def test_condition_needs_exactly_one_mode(self):
        for kwargs in ({}, dict(batch_size=16, batch_splits=2), dict(batch_splits=3), dict(batch_size=0)):
            with self.assertRaises(ValueError, msg=kwargs):
                BatchingCondition("x", **kwargs)


class OverlayTests(unittest.TestCase):
    def test_overlay_reproduces_the_measured_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            try:
                source = native_source(Path(temporary) / "native")
            except urllib.error.URLError as error:
                self.skipTest(f"PyPI unavailable: {error}")
            # build_overlay checks every file against the measured overlay hashes.
            overlay = build_overlay(source, Path(temporary) / "overlay")
            self.assertIn("criteria_per_batch", (overlay / "orchestrator.py").read_text())
            self.assertEqual((overlay / "judge.py").read_bytes(), (source / "judge.py").read_bytes())


@unittest.skipUnless((RELEASE_DIR / "shared-tools").is_dir(), "Run `make fetch` to download the release with shared tools")
class ConfigureTaskTests(unittest.TestCase):
    def _generate(self, root: Path) -> Path:
        release = BVBRelease.open(RELEASE_DIR)
        task = release.tasks[sorted(release.tasks)[0]]
        BVBAdapter(release, output_dir=root / "dataset", tool_root=release.root, verifier_model=MODEL,
                   reasoning_effort=REASONING_EFFORT, execution_environment="daytona",
                   daytona_snapshot=plan("gandalf-llm", release)["snapshot"]).generate([task])
        overlay = root / "overlay"
        overlay.mkdir()
        (overlay / "__init__.py").write_text("")
        return root / "dataset" / task.task_id

    def test_exact_size_condition_replaces_native_splits(self):
        with tempfile.TemporaryDirectory() as temporary:
            task_dir = self._generate(Path(temporary))
            configure_task(task_dir, BatchingCondition("b16", batch_size=16), Path(temporary) / "overlay")
            grader = tomllib.loads((task_dir / "tests/grader.toml").read_text())
            self.assertEqual(grader["batch_size"], 16)
            self.assertNotIn("batch_splits", grader)
            self.assertEqual(grader["max_concurrency"], 2)
            task = tomllib.loads((task_dir / "task.toml").read_text())
            self.assertEqual(task["verifier"]["timeout_sec"], VERIFIER_TIMEOUT_SEC)
            self.assertTrue((task_dir / "tests/gandalf/__init__.py").is_file())
            self.assertEqual((task_dir / "tests/bvb_batch_observe.py").read_bytes(),
                             (RUNTIME / "bvb_batch_observe.py").read_bytes())
            self.assertIn("[BVB_ACTUAL_SESSION]", (task_dir / "tests/sitecustomize.py").read_text())

    def test_split_condition_keeps_native_splits(self):
        with tempfile.TemporaryDirectory() as temporary:
            task_dir = self._generate(Path(temporary))
            configure_task(task_dir, BatchingCondition("split2", batch_splits=2), Path(temporary) / "overlay")
            grader = tomllib.loads((task_dir / "tests/grader.toml").read_text())
            self.assertEqual(grader["batch_splits"], 2)
            self.assertNotIn("batch_size", grader)


if __name__ == "__main__":
    unittest.main()

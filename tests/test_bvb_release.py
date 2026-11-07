"""Regression checks for the exact human-review judge input contract."""
import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'verifier-eval'))
from adapters.bvb.adapter import BVBAdapter, RewardKitBVBAdapter
from adapters.bvb.config import COHORT, DATASET_REVISION, DEFAULT_RELEASE_DIR
from adapters.bvb.contract import EXPECTED_MCP_TOOLS, REPLAY_EVIDENCE_GUIDANCE, REPLAY_VIEW, prepare_replay_view, canonical_rows, check_mcp_tools, input_identity, validate_rubric, validate_runtime
from adapters.bvb.release import BVBRelease, ReleaseError
from verifier_runner.inputs import resolve, validate_bundle
from verifier_runner.identity import stamp_criterion_identity
from verifier_runner.scoring import score_verifier, load_ground_truth
from verifier_runner.verdict import Verdict, CriterionVerdict

RELEASE = Path(os.environ.get('BVB_TEST_RELEASE', DEFAULT_RELEASE_DIR))

@unittest.skipUnless((RELEASE / 'shared-tools').is_dir(), 'Run `make fetch` to download the release with shared tools')
class ReleaseContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.release = BVBRelease.open(RELEASE)
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.task = cls.release.tasks['btb-515a104d']  # Two edited human criteria.
        cls.adapter = BVBAdapter(cls.release, output_dir=cls.root/'gandalf', tool_root=RELEASE)
        cls.generated = cls.adapter.generate([cls.task])[0].task_dir
        cls.rewardkit = RewardKitBVBAdapter(cls.release, output_dir=cls.root/'rewardkit', tool_root=RELEASE)
        cls.rk_generated = cls.rewardkit.generate([cls.task])[0].task_dir

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_release_roster_and_pin(self):
        result = self.release.validate_structure()
        self.assertEqual((result.task_count, result.adjudicated_task_count, result.eligible_criteria_count), (24, 24, 2464))
        self.assertEqual(self.release.revision, DATASET_REVISION)
        with self.assertRaises(ReleaseError):
            self.release.task_ids('adjudicated-20')

    def test_actual_text_checked_even_with_original_hash(self):
        rows = canonical_rows(self.task)
        altered = copy.deepcopy(rows)
        altered[0]['criterion'] += ' changed'
        with self.assertRaisesRegex(ValueError, 'field criterion'):
            validate_rubric(altered, rows)

    def test_all_selected_replay_identities_resolve_manifest_hashes(self):
        from adapters.bvb.contract import tree_hash, file_hash
        for task in self.release.tasks.values():
            with self.subTest(task=task.task_id):
                identity = input_identity(self.release, task)
                agent = self.release.root / task.selected_rollout.candidate_path / 'agent'
                self.assertEqual(identity['workspace_sha256'], tree_hash(agent / 'workspace'))
                self.assertEqual(identity['trajectory_sha256'], file_hash(agent / 'trajectory.json'))

    def test_weight_category_and_order_checked(self):
        rows = canonical_rows(self.task)
        for field in ['weight', 'category', 'source_criterion_index', 'criterion_text_sha256']:
            altered = copy.deepcopy(rows)
            altered[0][field] = 'wrong'
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_rubric(altered, rows)
        with self.assertRaises(ValueError):
            validate_rubric(rows[::-1], rows)

    def test_old_rubric_and_excluded_rows_rejected(self):
        original = json.loads((RELEASE/'task-source'/self.task.task_id/'tests/rubric.json').read_text())
        self.assertNotEqual(original[0]['criterion'], self.task.criteria[0].text)
        with self.assertRaises(ValueError):
            validate_rubric(original, canonical_rows(self.task))
        task = self.release.tasks['btb-067cb834']
        self.assertEqual(len(task.criteria), 104)
        self.assertEqual(len(self.release.canonical_labels(task)['labels']), 106)

    def test_scoring_verifies_final_annotation_files_without_workbooks(self):
        from scripts.score_bvb_verifier_against_humans import _verified_human_sources
        sources = _verified_human_sources(self.release, [self.task.task_id])
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]['path'], f'human-adjudication/canonical/{self.task.task_id}.json')
        self.assertEqual(sources[0]['status'], 'verified')
        self.assertNotIn('workbook', sources[0])

    def test_exact_prompt_and_authentic_replay_in_both_runtimes(self):
        for generated in [self.generated, self.rk_generated]:
            with self.subTest(generated=generated):
                self.assertEqual((generated/'instruction.md').read_text(), (RELEASE/'task-source'/self.task.task_id/'instruction.md').read_text())
                out = self.root / (generated.parent.name + '-audit.json')
                audit = validate_runtime(generated/'tests', generated/'solution/workspace', generated/'solution/trajectory.json', out)
                self.assertEqual(audit['identity'], input_identity(self.release, self.task))
                self.assertEqual(audit['criterion_count'], len(self.task.criteria))

    def test_full_replay_evidence_is_advertised_to_both_judges(self):
        for generated, name in [(self.generated, 'judge-guidance.md'), (self.rk_generated, 'judge-prompt.md')]:
            guidance = (generated / 'tests' / name).read_text()
            self.assertIn(REPLAY_EVIDENCE_GUIDANCE, guidance)
        with tempfile.TemporaryDirectory() as temporary:
            import shutil
            tests = Path(temporary) / 'tests'
            shutil.copytree(self.generated / 'tests', tests)
            path = tests / 'judge-guidance.md'
            path.write_text(path.read_text().replace(REPLAY_EVIDENCE_GUIDANCE, ''))
            with self.assertRaisesRegex(ValueError, 'guidance omits'):
                validate_runtime(tests, self.generated / 'solution/workspace', self.generated / 'solution/trajectory.json', Path(temporary) / 'audit.json')

    def test_symlink_view_keeps_replay_unchanged_and_checks_every_file(self):
        from adapters.bvb.contract import tree_hash, file_hash
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = self.generated / 'solution/workspace'
            trajectory = self.generated / 'solution/trajectory.json'
            before = (tree_hash(workspace), file_hash(trajectory))
            view = root / 'view'
            prepare_replay_view(workspace, trajectory, view)
            self.assertFalse((view / 'workspace').is_symlink())
            self.assertTrue((view / 'trajectory.json').is_symlink())
            for original in workspace.rglob('*'):
                if original.is_file():
                    self.assertTrue((view / 'workspace' / original.relative_to(workspace)).is_symlink())
            audit = validate_runtime(self.generated / 'tests', workspace, trajectory, root / 'audit.json', replay_view=view)
            self.assertTrue(audit['judge_view_validated'])
            self.assertEqual((tree_hash(workspace), file_hash(trajectory)), before)
            with self.assertRaises(FileExistsError):
                prepare_replay_view(workspace, trajectory, view)
            (view / 'trajectory.json').unlink()
            (view / 'trajectory.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'symlink view differs'):
                validate_runtime(self.generated / 'tests', workspace, trajectory, root / 'bad-audit.json', replay_view=view)

    def test_runtime_rejects_wrong_trajectory(self):
        bad = self.root/'bad-trajectory.json'
        bad.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'trajectory differs'):
            validate_runtime(self.generated/'tests', self.generated/'solution/workspace', bad, self.root/'bad-audit.json')

    def test_runtime_rejects_incomplete_workspace(self):
        empty = self.root/'empty-workspace'
        empty.mkdir(exist_ok=True)
        with self.assertRaisesRegex(ValueError, 'workspace differs'):
            validate_runtime(self.generated/'tests', empty, self.generated/'solution/trajectory.json', self.root/'empty-audit.json')

    def test_outer_audit_requires_mcp_probe_from_isolated_inner_role(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            probe_path = root / 'mcp-audit.json'
            probe = dict(status='validated', caller_uid=10003, caller_groups=[10002, 10003],
                         mcp_tools=sorted(EXPECTED_MCP_TOOLS))
            probe_path.write_text(json.dumps(probe))
            audit = validate_runtime(self.generated/'tests', self.generated/'solution/workspace',
                                     self.generated/'solution/trajectory.json', root/'audit.json',
                                     mcp_audit=probe_path)
            self.assertEqual(audit['mcp_probe'], probe)
            for changed in [dict(probe, caller_uid=10002),
                            dict(probe, caller_groups=[10001, 10002, 10003]),
                            dict(probe, mcp_tools=[])]:
                probe_path.write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError, 'isolated sandbox role'):
                    validate_runtime(self.generated/'tests', self.generated/'solution/workspace',
                                     self.generated/'solution/trajectory.json', root/'bad-audit.json',
                                     mcp_audit=probe_path)

    def test_generation_preserves_existing_output(self):
        before = (self.generated/'tests/rubric.json').read_bytes()
        with self.assertRaisesRegex(ReleaseError, 'overwrite'):
            self.adapter.generate([self.task])
        self.assertEqual(before, (self.generated/'tests/rubric.json').read_bytes())

    def test_host_checks_actual_bundle_after_resolution(self):
        bundle = resolve(COHORT, self.task.task_id, datasets_root=self.root, release_root=RELEASE, dataset_dir=self.generated.parent)
        bundle = replace(bundle, workspace_dir=self.generated/'solution/workspace', trajectory_path=self.generated/'solution/trajectory.json')
        validate_bundle(bundle)
        changed = copy.deepcopy(bundle.rubric)
        changed[0]['criterion'] += ' altered'
        with self.assertRaises(ValueError):
            validate_bundle(replace(bundle, rubric=changed))
        with self.assertRaises(ValueError):
            validate_bundle(replace(bundle, instructions=bundle.instructions+' extra'))

    def test_daytona_exports_actual_workspace_and_resolution_requires_collected_bytes(self):
        import shutil
        import tomllib
        from adapters.bvb.daytona import snapshot_name
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            adapter = BVBAdapter(self.release, output_dir=root/'generated', tool_root=RELEASE,
                                 execution_environment='daytona', daytona_snapshot=snapshot_name(self.release))
            generated = adapter.generate([self.task])[0].task_dir
            config = tomllib.loads((generated/'task.toml').read_text())
            self.assertEqual(config['artifacts'], [{'source':'/home/agent/workspace',
                                                   'destination':'reviewed-workspace'}])
            trial = root/'staging'/(self.task.task_id+'__remote')
            (trial/'agent').mkdir(parents=True)
            shutil.copy2(generated/'solution/trajectory.json',trial/'agent/trajectory.json')
            kwargs = dict(datasets_root=root, release_root=RELEASE,
                          dataset_dir=generated.parent, staging_jobs_dir=trial.parent)
            with self.assertRaisesRegex(ValueError, 'lacks authentic staged workspace'):
                resolve(COHORT, self.task.task_id, **kwargs)
            exported = trial/'artifacts/reviewed-workspace'
            shutil.copytree(generated/'solution/workspace',exported)
            bundle = resolve(COHORT, self.task.task_id, **kwargs)
            self.assertEqual(bundle.workspace_dir,exported)
            validate_bundle(bundle)
            (exported/'extra.txt').write_text('changed after grading')
            with self.assertRaisesRegex(ValueError, 'staged replay differs'):
                resolve(COHORT, self.task.task_id, **kwargs)

    def test_evaluation_persists_actual_inputs_and_rejects_overwrite(self):
        from verifier_runner.evaluate import evaluate
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trial = root / 'staging' / (self.task.task_id + '__test')
            (trial / 'agent').mkdir(parents=True)
            (trial / 'verifier').mkdir()
            (trial / 'agent/workspace').symlink_to(self.generated / 'solution/workspace', target_is_directory=True)
            (trial / 'agent/trajectory.json').symlink_to(self.generated / 'solution/trajectory.json')
            identity = input_identity(self.release, self.task)
            (trial / 'verifier/input-audit.json').write_text(json.dumps({'status': 'validated', 'identity': identity, 'mcp_tools': sorted(EXPECTED_MCP_TOOLS), 'judge_workdir': REPLAY_VIEW, 'judge_view_validated': True}))
            rows = [dict(criterion=c.text, weight=c.weight, met=True, reasoning='test') for c in self.task.criteria]
            (trial / 'verifier/info.json').write_text(json.dumps({'criterion_results': rows}))
            kwargs = dict(verifiers=['gandalf'], cohort=COHORT,
                          staging_jobs_dir=trial.parent, results_dir=root / 'results', run_id='test',
                          datasets_root=self.root, release_root=RELEASE,
                          tasks=[self.task.task_id], dataset_dir=self.generated.parent)
            result = evaluate(**kwargs)
            self.assertEqual(len(result.verdicts_written), 1)
            self.assertEqual(result.errors, [])
            verdict_path = result.verdicts_written[0]
            verdict = Verdict.model_validate_json(verdict_path.read_text())
            self.assertEqual(verdict.input_identity, identity)
            request = json.loads((root / 'results/test/traces/gandalf' / self.task.task_id / 'judge-inputs.json').read_text())
            self.assertEqual(request['rubric'], canonical_rows(self.task))
            previous = verdict_path.read_bytes()
            with self.assertRaises(FileExistsError):
                evaluate(**kwargs)
            self.assertEqual(verdict_path.read_bytes(), previous)

    def test_lazy_mcp_configuration_survives_filtered_judge_environment(self):
        import tomllib
        server = tomllib.loads((self.generated / 'tests/grader.toml').read_text())['mcp_servers'][0]
        self.assertEqual(server['command'], '/usr/bin/env')
        self.assertIn('BVB_TOOL_ARCHIVE_INDEX=/opt/mcp-server/data/tool-archive-index.json', server['args'])
        rewardkit = json.loads((self.rk_generated / 'tests/mcp-config.json').read_text())
        self.assertEqual(server, rewardkit)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            script = root / 'stdio_server.py'
            script.write_text('import json, os, sys\n'
                              'for line in sys.stdin:\n'
                              ' m=json.loads(line)\n'
                              ' if m.get("method")=="tools/list":\n'
                              f'  names={sorted(EXPECTED_MCP_TOOLS)!r} if os.environ.get("READY")=="1" else []\n'
                              '  print(json.dumps({"jsonrpc":"2.0","id":m["id"],"result":{"tools":[{"name":n} for n in names]}}), flush=True)\n')
            configured = dict(command='/usr/bin/env', args=['READY=1', sys.executable, str(script)])
            (root / 'mcp-config.json').write_text(json.dumps(configured))
            self.assertEqual(set(check_mcp_tools(root)), EXPECTED_MCP_TOOLS)
            # The inner probe must work without any original replay files.
            import subprocess
            subprocess.run([sys.executable, str(ROOT/'adapters/bvb/contract.py'),
                            '--tests', str(root), '--check-mcp-only',
                            '--output', str(root/'mcp-audit.json')], check=True, capture_output=True)
            self.assertEqual(set(json.loads((root/'mcp-audit.json').read_text())['mcp_tools']), EXPECTED_MCP_TOOLS)
            configured['args'] = [sys.executable, str(script)]
            (root / 'mcp-config.json').write_text(json.dumps(configured))
            with self.assertRaisesRegex(ValueError, 'MCP tools differ'):
                check_mcp_tools(root)

    def test_complete_cohort_scoring_and_overall_scores(self):
        verdicts = []
        for task in self.release.tasks.values():
            labels = load_ground_truth(self.release, task.task_id)
            verdicts.append(Verdict(verifier='gandalf', model='test', task_id=task.task_id, cohort=COHORT,
                canonical_rubric_sha256=task.manifest['evaluation_rubric']['sha256'],
                input_identity=input_identity(self.release, task),
                criteria=[CriterionVerdict(index=i, evaluation_criterion_index=i,
                    source_criterion_index=c.source_index, criterion_text_sha256=c.text_sha256,
                    satisfied=labels[i]['decision']=='MET') for i,c in enumerate(task.criteria)]))
        score = score_verifier(verdicts, RELEASE, cohort=COHORT)
        self.assertEqual(score.metrics.counts.total, 2464)
        self.assertEqual(score.metrics.accuracy, 1)
        self.assertTrue(all(t.overall_score_difference == 0 for t in score.per_task))
        verdicts[0].input_identity = None
        with self.assertRaisesRegex(ValueError, 'input identity'):
            score_verifier(verdicts, RELEASE, cohort=COHORT)

if __name__ == '__main__':
    unittest.main()

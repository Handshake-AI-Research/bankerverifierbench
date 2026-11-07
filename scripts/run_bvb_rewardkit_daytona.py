#!/usr/bin/env python3
"""Stage native RewardKit/Codex runs in a verified, separate Daytona snapshot."""
import argparse
import asyncio
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "verifier-eval")]
from dotenv import load_dotenv
from adapters.bvb.config import COHORT, DEFAULT_RELEASE_DIR
from adapters.bvb.adapter import RewardKitBVBAdapter
from adapters.bvb.release import BVBRelease
from adapters.bvb.daytona import _snapshot_identity, _probe_snapshot_sandbox, _python_stdin_command
from adapters.bvb.runtime_snapshots import plan as runtime_plan
from verifier_runner.stage import stage
from verifier_runner.inputs import resolve, validate_bundle, list_staged_trials
from verifier_runner.runners.rewardkit import invoke
from verifier_runner.identity import stamp_criterion_identity
from verifier_runner.scoring import _validate_verdict_identity
from verifier_runner.runners._codex_audit import audit_sessions
from verifier_runner.verdict import TokenUsage
from adapters.bvb.contract import EXPECTED_MCP_TOOLS
from types import SimpleNamespace


async def preflight_snapshot(plan, expected):
    from daytona import AsyncDaytona, CreateSandboxFromSnapshotParams
    async with AsyncDaytona() as daytona:
        sandbox = await daytona.create(CreateSandboxFromSnapshotParams(
            snapshot=plan['snapshot'], ephemeral=True), timeout=120)
        try:
            result = await _probe_snapshot_sandbox(sandbox, snapshot=plan['snapshot'], expected_identity=expected)
            probe = await sandbox.process.exec(_python_stdin_command('''
import hashlib, json
from pathlib import Path
identity = json.loads(Path('/opt/bvb-rewardkit-identity.json').read_text())
identity['codex_wrapper_sha256'] = hashlib.sha256(Path('/usr/local/bin/codex').read_bytes()).hexdigest()
print(json.dumps(identity))
'''), timeout=60)
            if probe.exit_code or json.loads(probe.result.strip()) != plan['identity']:
                raise ValueError('RewardKit runtime identity or CLI wrapper hash differs from snapshot plan')
            fix = plan['identity'].get('logo_export_permissions')
            if fix:
                source = "import hashlib; from pathlib import Path; print(hashlib.sha256(Path(" + repr(fix['path']) + ").read_bytes()).hexdigest())"
                probe = await sandbox.process.exec(_python_stdin_command(source), timeout=30)
                if probe.exit_code or probe.result.strip()!=fix['patched_sha256']:
                    raise ValueError('MCP exported-logo permission fix differs from snapshot plan')
            cli = await sandbox.process.exec('codex --version', timeout=60)
            if cli.exit_code or cli.result.strip() != 'codex-cli ' + plan['identity']['codex_version']:
                raise ValueError('RewardKit Codex CLI version differs from snapshot plan')
            result['rewardkit_runtime'] = plan['identity']
            return result
        finally:
            await sandbox.delete()


def collect(a, release, tasks, dataset, result, job_id):
    repeat_start = getattr(a, 'repeat_start', 1)
    summary = dict(job_id=job_id, stage_returncode=result.returncode, model=a.model,
                   reasoning_effort=a.reasoning_effort, tasks=len(tasks),
                   attempts=a.attempts, repeat_start=repeat_start, runs=[], pending=[], issues=[])
    for task in tasks:
        candidates=list_staged_trials(result.jobs_dir, task)
        # Harbor creates random-name trial directories incrementally. Their
        # sorted positions are stable only after the task's complete triplet.
        # Keep early observations out of repetition-indexed comparison files.
        if a.collect_only and (len(candidates)!=a.attempts or
                any(not (trial/'result.json').exists() for trial in candidates)):
            summary['pending'].extend(dict(task_id=task,repeat=r+repeat_start) for r in range(a.attempts))
            continue
        if len(candidates)!=a.attempts:
            summary['issues'].append(dict(task_id=task,error='Unexpected number of native trial directories',
                                          expected=a.attempts,actual=len(candidates)))
            continue
        for repeat in range(a.attempts):
            try:
                bundle = resolve(COHORT,task,datasets_root=ROOT / "datasets",
                    dataset_dir=dataset, release_root=release.root,
                    staging_jobs_dir=result.jobs_dir, trial_index=repeat, verifier_kind="rewardkit")
                native_path=bundle.trial_dir / "result.json"
                if a.collect_only and not native_path.exists():
                    summary['pending'].append(dict(task_id=task,repeat=repeat+repeat_start))
                    continue
                native = json.loads(native_path.read_text())
                if native.get("exception_info"):
                    raise ValueError("Harbor trial records an exception")
                trace = a.output / "rewardkit" / a.model.replace("/", "__") / f"repeat-{repeat+repeat_start}" / "traces" / task
                target = trace.parents[1] / "verdicts" / (task + ".json")
                if (bundle.trial_dir/'verifier/excluded-from-comparison.json').exists():
                    if target.exists() and not (trace/'excluded-from-comparison.json').exists():
                        with (trace/'excluded-from-comparison.json').open('x') as stream:
                            json.dump(dict(reason='Native attempt failed follow-up audit',
                                           native_trial=str(bundle.trial_dir.resolve())),stream,indent=2)
                    raise ValueError('Native attempt is excluded by its follow-up audit')
                if target.exists():
                    binding=trace/'native-trial-identity.json'
                    if not binding.exists() or json.loads(binding.read_text())['trial']!=str(bundle.trial_dir.resolve()):
                        raise ValueError('Existing normalized repetition is not bound to this native trial')
                    summary['runs'].append(dict(task_id=task, repeat=repeat+repeat_start, status='already_collected'))
                    continue
                verdict = stamp_criterion_identity(invoke(bundle, trace_dir=trace),bundle)
                after = json.loads((trace / 'input-audit-after.json').read_text())
                if after.get('status') != 'validated' or after.get('identity') != bundle.input_identity:
                    raise ValueError('RewardKit post-run evidence identity audit failed')
                from verifier_runner.runners._codex_audit import deliverable_evidence_names
                audit_name = ('codex-audit-recovery-' + datetime.now().strftime('%Y%m%dT%H%M%S') + '.json'
                              if (trace/'codex-audit.json').exists() else 'codex-audit.json')
                audit = audit_sessions(trace, a.model.removeprefix('openai/'), a.reasoning_effort,
                    deliverable_evidence_names(bundle.workspace_dir), EXPECTED_MCP_TOOLS,
                    bundle.instructions, [c['criterion'] for c in bundle.rubric],audit_name=audit_name)
                verdict.tokens = TokenUsage(**audit['tokens'])
                if audit['model_cost_usd'] is not None: verdict.cost_usd = audit['model_cost_usd']
                _validate_verdict_identity(release, task, verdict); validate_bundle(bundle)
                timing = native["verifier"]
                verdict.duration_sec = (datetime.fromisoformat(timing["finished_at"].replace("Z", "+00:00"))-
                                        datetime.fromisoformat(timing["started_at"].replace("Z", "+00:00"))).total_seconds()
                target.parent.mkdir(exist_ok=True)
                with target.open("x") as stream:stream.write(verdict.model_dump_json(indent=2))
                with (trace/'native-trial-identity.json').open('x') as stream:
                    json.dump(dict(trial=str(bundle.trial_dir.resolve()),task_id=task,repeat=repeat+repeat_start,
                                   native_trial_index=repeat),stream,indent=2)
                summary["runs"].append(dict(task_id=task,repeat=repeat+repeat_start,trial=str(bundle.trial_dir),
                     criteria=len(verdict.criteria),duration_sec=verdict.duration_sec, flags=audit['flags']))
            except Exception as error:
                summary['issues'].append(dict(task_id=task, repeat=repeat+repeat_start,
                    error_type=type(error).__name__, error=str(error)))
    name = ('collection-' + datetime.now().strftime('%Y%m%dT%H%M%S') + '.json') if a.collect_only else 'completion.json'
    with (a.output / name).open('x') as stream:json.dump(summary, stream, indent=2)
    print(json.dumps(dict(collected=len(summary['runs']), issues=len(summary['issues']), output=str(a.output/name))),flush=True)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", default="openai/gpt-5.4")
    p.add_argument("--tasks", default="")
    p.add_argument("--attempts", type=int, default=3)
    p.add_argument("--repeat-start", type=int, default=1,
                   help="Benchmark repetition slot for targeted retries; native trial indices remain independent")
    p.add_argument("--concurrency", type=int, default=25)
    p.add_argument("--reasoning-effort", default="high")
    p.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE_DIR)
    p.add_argument('--collect-only', action='store_true', help='Audit completed native trials without launching model requests or replacing results')
    p.add_argument("--snapshot-plan", type=Path,
                   help="Derived runtime plan JSON (default: computed rewardkit plan for the pinned release)")
    a = p.parse_args()
    if a.repeat_start < 1 or a.attempts < 1 or a.repeat_start + a.attempts - 1 > 3:
        p.error('Requested repetitions must fit the three benchmark slots')
    load_dotenv(ROOT / ".env")
    a.output = a.output.resolve()
    if a.collect_only:
        if not (a.output/'runtime-preflight.json').exists():raise ValueError('Existing condition lacks runtime preflight')
    else:a.output.mkdir(parents=True, exist_ok=False)
    release = BVBRelease.open(a.release_dir)
    plan = json.loads(a.snapshot_plan.read_text()) if a.snapshot_plan else runtime_plan("rewardkit", release)
    tasks = a.tasks.split(",") if a.tasks else sorted(release.tasks)
    dataset = a.output / "dataset"/COHORT
    job_id = "bvb-rk-" + a.output.name + "-" + a.model.replace("/", "-")
    if a.collect_only:
        result=SimpleNamespace(returncode=None,jobs_dir=ROOT/'jobs'/job_id)
        collect(a,release,tasks,dataset,result,job_id)
        return 0
    expected = _snapshot_identity(release, release_revision=release.revision, cache_max_bytes=2147483648)
    preflight = asyncio.run(preflight_snapshot(plan, expected))
    (a.output / "runtime-preflight.json").write_text(json.dumps(preflight, indent=2))
    adapter = RewardKitBVBAdapter(release, output_dir=a.output / "dataset"/COHORT,
        tool_root=release.root, verifier_model=a.model, rewardkit_agent="codex",
        rewardkit_reasoning_effort=a.reasoning_effort, execution_environment="daytona",
        daytona_snapshot=plan["snapshot"])
    adapter.generate([release.tasks[task] for task in tasks])
    result = stage(cohort=COHORT, tasks=tasks, staging_id=job_id,
        repo_root=ROOT, dataset_dir=dataset, verifier_kind="rewardkit",
        environment_type="daytona", n_concurrent=a.concurrency, n_attempts=a.attempts,
        max_retries=2, environment_kwargs=["snapshot_template_name=" + plan["snapshot"]])
    summary=collect(a,release,tasks,dataset,result,job_id)
    return result.returncode or int(bool(summary['issues']))


if __name__ == "__main__":
    raise SystemExit(main())

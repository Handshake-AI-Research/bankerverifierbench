#!/usr/bin/env python3
"""Run model ablations with a separately verified Gandalf transport runtime."""
import argparse,asyncio,json
from datetime import datetime
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'verifier-eval')]
from dotenv import load_dotenv
from adapters.bvb.config import COHORT, DEFAULT_RELEASE_DIR
from adapters.bvb.adapter import BVBAdapter,effective_reasoning_config
from adapters.bvb.release import BVBRelease
from adapters.bvb.daytona import _snapshot_identity,_probe_snapshot_sandbox,_python_stdin_command
from adapters.bvb.runtime_snapshots import plan as runtime_plan
from verifier_runner.stage import stage
from verifier_runner.inputs import resolve,validate_bundle
from verifier_runner.runners.gandalf import invoke
from verifier_runner.identity import stamp_criterion_identity
from verifier_runner.scoring import _validate_verdict_identity
from scripts.run_bvb_gandalf_daytona import validate_job

async def preflight(plan,release):
 from daytona import AsyncDaytona,CreateSandboxFromSnapshotParams
 async with AsyncDaytona() as d:
  s=await d.create(CreateSandboxFromSnapshotParams(snapshot=plan['snapshot'],ephemeral=True),timeout=120)
  try:
   result=await _probe_snapshot_sandbox(s,snapshot=plan['snapshot'],expected_identity=_snapshot_identity(release,release_revision=release.revision,cache_max_bytes=2147483648))
   r=await s.process.exec(_python_stdin_command("import json; from pathlib import Path; print(Path('/opt/bvb-model-runtime-identity.json').read_text())"),timeout=60)
   if r.exit_code or json.loads(r.result)!=plan['identity']:raise ValueError('Ablation snapshot identity differs')
   fix=plan['identity'].get('logo_export_permissions')
   if fix:
    r=await s.process.exec('sha256sum '+fix['path'],timeout=30)
    if r.exit_code or r.result.split()[0]!=fix['patched_sha256']:raise ValueError('MCP logo export fix differs')
   capability_fix=plan['identity'].get('sdk_reasoning_capability_backport')
   if capability_fix:
    r=await s.process.exec('sha256sum '+capability_fix['path'],timeout=30)
    if r.exit_code or r.result.split()[0]!=capability_fix['patched_sha256']:raise ValueError('SDK reasoning capability backport differs')
   source="import json,importlib.metadata as m;print(json.dumps({p:m.version(p) for p in ['gandalf-the-grader','openhands-sdk','openhands-tools','litellm']}))"
   import shlex
   r=await s.process.exec('/opt/uv-tools/gandalf-the-grader/bin/python -c '+shlex.quote(source),timeout=60)
   if r.exit_code or json.loads(r.result)!=plan['identity']['packages']:raise ValueError('Ablation package versions differ')
   result['model_runtime']=plan['identity'];return result
  finally:await s.delete()

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--model',required=True)
 p.add_argument('--tasks',default='');p.add_argument('--attempts',type=int,default=3);p.add_argument('--concurrency',type=int,default=25)
 p.add_argument('--repeat-start',type=int,default=1,help='First benchmark repetition slot; native trial indices remain independent')
 p.add_argument('--reasoning-effort',default='high');p.add_argument('--release-dir',type=Path,default=DEFAULT_RELEASE_DIR);p.add_argument('--snapshot-plan',type=Path,help='Derived runtime plan JSON (default: computed gandalf-llm plan for the pinned release)')
 a=p.parse_args()
 if a.repeat_start<1 or a.attempts<1 or a.repeat_start+a.attempts-1>3:p.error('Requested repetitions must fit benchmark slots 1 through 3')
 load_dotenv(ROOT/'.env');a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
 release=BVBRelease.open(a.release_dir);plan=json.loads(a.snapshot_plan.read_text()) if a.snapshot_plan else runtime_plan('gandalf-llm',release)
 (a.output/'runtime-preflight.json').write_text(json.dumps(asyncio.run(preflight(plan,release)),indent=2))
 tasks=a.tasks.split(',') if a.tasks else sorted(release.tasks);dataset=a.output/'dataset'/COHORT
 BVBAdapter(release,output_dir=dataset,tool_root=release.root,verifier_model=a.model,reasoning_effort=a.reasoning_effort,
     execution_environment='daytona',daytona_snapshot=plan['snapshot']).generate([release.tasks[t] for t in tasks])
 job='bvb-gd-'+a.output.name+'-'+a.model.replace('/','-')
 result=stage(cohort=COHORT,tasks=tasks,staging_id=job,repo_root=ROOT,dataset_dir=dataset,
     environment_type='daytona',n_concurrent=a.concurrency,n_attempts=a.attempts,max_retries=2,
     retry_include=['DaytonaError','DaytonaConnectionError'],environment_kwargs=['snapshot_template_name='+plan['snapshot']])
 if result.returncode:raise RuntimeError('Harbor Gandalf stage failed')
 summary=validate_job(result.jobs_dir,task_ids=tasks,attempts=a.attempts,verifier_model=a.model,
     reasoning_config=effective_reasoning_config(a.reasoning_effort),release_dir=release.root)
 for task in tasks:
  for repeat in range(a.attempts):
   b=resolve(COHORT,task,datasets_root=ROOT/'datasets',dataset_dir=dataset,release_root=release.root,
       staging_jobs_dir=result.jobs_dir,trial_index=repeat)
   benchmark_repeat=repeat+a.repeat_start
   trace=a.output/'gandalf'/a.model.replace('/','__')/f'repeat-{benchmark_repeat}'/'traces'/task
   v=stamp_criterion_identity(invoke(b,trace_dir=trace),b);_validate_verdict_identity(release,task,v);validate_bundle(b)
   native=json.loads((b.trial_dir/'result.json').read_text());timing=native['verifier']
   v.duration_sec=(datetime.fromisoformat(timing['finished_at'])-datetime.fromisoformat(timing['started_at'])).total_seconds()
   target=trace.parents[1]/'verdicts'/(task+'.json');target.parent.mkdir(exist_ok=True)
   with target.open('x') as f:f.write(v.model_dump_json(indent=2))
   with (trace/'native-trial-identity.json').open('x') as f:json.dump(dict(trial=str(b.trial_dir.resolve()),task_id=task,repeat=benchmark_repeat,native_trial_index=repeat),f,indent=2)
 summary.update(model=a.model,reasoning_effort=a.reasoning_effort,repeat_start=a.repeat_start)
 (a.output/'completion.json').write_text(json.dumps(summary,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())

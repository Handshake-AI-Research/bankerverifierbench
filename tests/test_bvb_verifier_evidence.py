import asyncio
import importlib.util
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "verifier-eval"))
from verifier_runner.runners.autorubric import _build_to_grade
from verifier_runner.runners.archipelago import _atif_to_archipelago_trajectory
from verifier_runner.runners._run_audit import criterion_errors, call_audit, reader_audit
from verifier_runner.runners._codex_audit import audit_sessions, _mcp_failed, deliverable_evidence_names


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EvidenceTests(unittest.TestCase):
    def test_deliverable_alias_requires_identical_bytes_and_excludes_source_files(self):
        workspace = Path(self.tmp.name)/'workspace'
        output = workspace/'banker_workspace/deliverables';output.mkdir(parents=True)
        (output/'graded.xlsx').write_bytes(b'graded workbook')
        (workspace/'root-alias.xlsx').write_bytes(b'graded workbook')
        (workspace/'unrelated.xlsx').write_bytes(b'source workbook')
        self.assertEqual(deliverable_evidence_names(workspace), ['graded.xlsx','root-alias.xlsx'])

    def test_rewardkit_missing_explanations_cannot_be_accepted_as_complete_grades(self):
        from verifier_runner.runners.rewardkit import _criteria_from_details
        rubric=[dict(criterion='exact criterion')]
        details={'rubric':{'criteria':[dict(description='exact criterion',value=1,reasoning='')]}}
        with self.assertRaisesRegex(ValueError, 'no grading explanation'):
            _criteria_from_details(details,rubric,require_explanations=True)
        self.assertTrue(_criteria_from_details(details,rubric)[0].satisfied)
        details['rubric']['criteria'][0]['reasoning']='Observed workbook cell evidence'
        self.assertTrue(_criteria_from_details(details,rubric,require_explanations=True)[0].satisfied)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "trajectory.json"
        self.steps = [dict(step_id=1, source="agent", tool_calls=[dict(
            function_name="read_file", arguments={"path": "deliverable.xlsx"})],
            observation={"results": [{"content": "independent_observation"}]}),
            dict(step_id=2, source="agent", message="final_response")]
        self.path.write_text(json.dumps({"steps": self.steps}))

    def tearDown(self):
        self.tmp.cleanup()

    def test_autorubric_retains_tool_only_steps_and_observations(self):
        text = _build_to_grade(self.path)
        self.assertIn("deliverable.xlsx", text)
        self.assertIn("independent_observation", text)
        self.assertIn("final_response", text)

    def test_aaaj_conversion_preserves_native_evidence_fields(self):
        worker = load(ROOT / "verifier-eval/verifiers/aaaj/worker.py", "aaaj_evidence_worker")
        steps = worker._atif_to_aaaj_trajectory(self.path)
        self.assertEqual(len(steps), 2)
        self.assertIn("deliverable.xlsx", steps[0]["agent"]["action"])
        self.assertIn("independent_observation", steps[0]["environment"])

    def test_archipelago_preserves_prompt_and_final_text(self):
        result = _atif_to_archipelago_trajectory(self.path, "frozen_task_prompt")
        self.assertEqual(result["messages"][0]["content"], "frozen_task_prompt")
        self.assertIn("independent_observation", result["messages"][1]["content"])
        self.assertEqual(result["output"]["text"], "final_response")

    def test_async_trace_tags_do_not_leak_across_criteria(self):
        trace = load(ROOT / "verifier-eval/verifiers/_shared/litellm_trace.py", "trace_tags_test")
        async def sample(index):
            with trace.tag(criterion_index=index):
                await asyncio.sleep(0)
                return trace._current_tags()["criterion_index"]
        async def both():
            return await asyncio.gather(sample(0), sample(1))
        self.assertEqual(asyncio.run(both()), [0, 1])

    def test_native_error_fallback_cannot_be_scored_as_a_defect(self):
        rows = [dict(index=0, reason='default: Error parsing judge response: litellm.RateLimitError: quota exceeded'),
                dict(index=1, reason='The trajectory contains an error, but the deliverable is correct.')]
        self.assertEqual([x['index'] for x in criterion_errors(rows)], [0])

    def test_reader_warning_is_visible_despite_successful_grading(self):
        trace=Path(self.tmp.name)
        (trace/'worker-stderr.txt').write_text("WARNING Reducto extraction failed for filesystem/broken.xlsx: 415\n"
            "[JUDGE][DIFF][METRICS] Reducto API calls: 25 (success: 24, failed: 1)\n")
        audit=reader_audit(trace)
        self.assertTrue(audit['requires_evidence_review'])
        self.assertEqual(audit['reducto_metrics'][0]['failed'],1)

    def test_mcp_application_error_is_not_a_successful_source_lookup(self):
        self.assertTrue(_mcp_failed(dict(isError=False,content=[dict(type='text',text='{"success":false,"error":"not found"}')])))
        self.assertTrue(_mcp_failed(dict(isError=True)))
        self.assertFalse(_mcp_failed(dict(isError=False,content=[dict(type='text',text='{"success":true,"result":"source evidence"}')])))

    def test_codex_audit_checks_actual_inputs_and_deduplicates_usage(self):
        trace = Path(self.tmp.name) / 'trace'
        sessions = trace / 'judge_trajectories/sessions'
        sessions.mkdir(parents=True)
        def row(kind, payload): return dict(type=kind, payload=payload)
        tokens = row('event_msg', dict(type='token_count', info=dict(
            total_token_usage=dict(total_tokens=110), last_token_usage=dict(
                input_tokens=100, cached_input_tokens=20, output_tokens=10))))
        rows = [row('turn_context', dict(model='gpt-5.4', effort='high')),
                row('response_item', dict(type='message', role='user', content=[dict(type='input_text', text='frozen prompt; exact criterion')])),
                row('response_item', dict(type='function_call', call_id='read', name='exec_command', arguments='openpyxl.load_workbook("deliverable.xlsx")')),
                row('response_item', dict(type='function_call_output', call_id='read', output='Process exited with code 0\nworksheet values')),
                row('response_item', dict(type='function_call', call_id='lookup', name='get_company_data', arguments='{}')),
                row('response_item', dict(type='function_call_output', call_id='lookup', output='source information')),
                tokens, tokens]
        path = sessions / 'session.jsonl'
        path.write_text('\n'.join(json.dumps(x) for x in rows))
        audit = audit_sessions(trace, 'gpt-5.4', 'high', ['deliverable.xlsx'], {'get_company_data'},
                               'frozen prompt', ['exact criterion'])
        self.assertEqual(audit['usage_requests'], 1)
        self.assertAlmostEqual(audit['model_cost_usd'], .000355)
        self.assertEqual(audit['successful_deliverable_commands'], 1)
        with self.assertRaisesRegex(ValueError, 'Canonical rubric text missing'):
            audit_sessions(trace, 'gpt-5.4', 'high', ['deliverable.xlsx'], {'get_company_data'},
                           'frozen prompt', ['different criterion'])

    def test_codex_code_mode_captures_returned_file_and_mcp_evidence(self):
        trace=Path(self.tmp.name)/'code-mode';sessions=trace/'judge_trajectories/sessions';sessions.mkdir(parents=True)
        result=dict(exit_code=0,output='worksheet values')
        mcp=dict(content=[dict(type='text',text='{"success":true,"result":"independent source"}')])
        source='text(await tools.exec_command({cmd:"openpyxl.load_workbook(\\\"deliverable.xlsx\\\")"})); text(await tools.mcp__bvb__get_company_data({}));'
        rows=[dict(type='turn_context',payload=dict(model='gpt-6.1-sol',effort='high')),
              dict(type='response_item',payload=dict(type='message',role='user',content=[dict(type='input_text',text='frozen prompt; exact criterion')])),
              dict(type='response_item',payload=dict(type='custom_tool_call',call_id='read',name='exec',input=source)),
              dict(type='response_item',payload=dict(type='custom_tool_call_output',call_id='read',output=[
                  dict(type='input_text',text=json.dumps(result)),
                  dict(type='input_text',text=json.dumps(dict(i=1,result=dict(status='fulfilled',value=mcp))))]))]
        (sessions/'session.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
        audit=audit_sessions(trace,'gpt-6.1-sol','high',['deliverable.xlsx'],{'get_company_data'},'frozen prompt',['exact criterion'])
        self.assertEqual(audit['successful_deliverable_commands'],1)
        self.assertEqual(audit['successful_mcp_calls'],1)
        self.assertEqual(audit['code_mode_wrapper_calls'],1)

    def test_code_mode_success_does_not_conceal_failed_or_opaque_reads(self):
        from verifier_runner.runners._codex_audit import _code_mode_evidence
        call=dict(input='text(await tools.exec_command({cmd:"openpyxl.load_workbook(\\\"deliverable.xlsx\\\")"})); text(await tools.mcp__bvb__get_company_data({}));')
        outputs=[dict(type='input_text',text=json.dumps(dict(exit_code=0,output='pwd'))),
                 dict(type='input_text',text=json.dumps(dict(exit_code=1,output='permission denied /home/verifier/workspace/source.xlsx'))),
                 dict(type='input_text',text=json.dumps(dict(content=[dict(type='text',text='{"success":false}')])))]
        files,mcp,failures,fatal=_code_mode_evidence(call,outputs,['deliverable.xlsx'],{'get_company_data'})
        self.assertEqual((files,mcp),(0,0));self.assertEqual(len(failures),2);self.assertTrue(fatal)
        files,*_=_code_mode_evidence(call,[dict(type='input_text',text='Script completed')],['deliverable.xlsx'],set())
        self.assertEqual(files,0)

    def test_code_mode_glob_read_can_identify_deliverable_in_returned_stdout(self):
        from verifier_runner.runners._codex_audit import _code_mode_evidence
        call=dict(input='text(await tools.exec_command({cmd:"import openpyxl; p=glob.glob(\\\"workspace/banker_workspace/deliverables/*\\\")[0]; w=openpyxl.load_workbook(p); print(p,w.sheetnames)"}));')
        returned=[dict(type='input_text',text=json.dumps(dict(exit_code=0,output="deliverable.xlsx ['Sheet1']")))]
        files,*_=_code_mode_evidence(call,returned,['deliverable.xlsx'],set())
        self.assertEqual(files,1)
        files,*_=_code_mode_evidence(call,returned,['different.xlsx'],set())
        self.assertEqual(files,0)

    def test_host_input_audit_uses_messages_instead_of_criterion_tags(self):
        trace = Path(self.tmp.name) / 'host-trace'; trace.mkdir()
        (trace/'judge-inputs.json').write_text(json.dumps(dict(instructions='frozen task prompt',
            rubric=[dict(criterion='edited human criterion')])))
        call=dict(model='gemini/gemini-3-flash-preview', metadata=dict(criterion_index=0, criterion='edited human criterion'),
                  messages=[dict(role='user',content='frozen task prompt; obsolete criterion')],
                  response=dict(choices=[dict(finish_reason='stop')]),cost_usd=.01)
        path=trace/'llm_calls.jsonl';path.write_text(json.dumps(call)+'\n')
        self.assertEqual(call_audit(trace)['criteria_missing_from_actual_messages'],[0])
        call['messages'][0]['content']='frozen task prompt; edited human criterion'
        path.write_text(json.dumps(call)+'\n')
        self.assertEqual(call_audit(trace)['criteria_missing_from_actual_messages'],[])

    def test_gpt6_cache_writes_use_replacement_rate_and_reasoning_is_not_double_counted(self):
        trace=Path(self.tmp.name)/'cache-cost';sessions=trace/'judge_trajectories/sessions';sessions.mkdir(parents=True)
        rows=[dict(type='turn_context',payload=dict(model='gpt-6-sol',effort='high')),
              dict(type='response_item',payload=dict(type='function_call',call_id='read',name='exec_command',arguments='openpyxl.load_workbook("deliverable.xlsx")')),
              dict(type='response_item',payload=dict(type='function_call_output',call_id='read',output='Process exited with code 0')),
              dict(type='event_msg',payload=dict(type='token_count',info=dict(total_token_usage=dict(total_tokens=110),
                   last_token_usage=dict(input_tokens=100,cached_input_tokens=20,cache_write_input_tokens=30,
                                         output_tokens=10,reasoning_output_tokens=3))))]
        (sessions/'session.jsonl').write_text('\n'.join(json.dumps(x) for x in rows))
        audit=audit_sessions(trace,'gpt-6-sol','high',['deliverable.xlsx'],set())
        self.assertAlmostEqual(audit['model_cost_usd'],.000279)
        self.assertEqual(audit['tokens']['cache_write_prompt_tokens'],30)
        self.assertEqual(audit['tokens']['completion_tokens'],10)
        self.assertEqual(audit['tokens']['reasoning_completion_tokens'],3)

    def test_partial_native_triplet_is_not_assigned_moving_repeat_numbers(self):
        from scripts.run_bvb_rewardkit_daytona import collect
        from types import SimpleNamespace
        root=Path(self.tmp.name);jobs=root/'jobs';trial=jobs/'btb-example__z';trial.mkdir(parents=True)
        (trial/'result.json').write_text('{}')
        out=root/'output';out.mkdir()
        args=SimpleNamespace(model='openai/gpt-5.4',reasoning_effort='high',attempts=3,collect_only=True,output=out)
        result=SimpleNamespace(returncode=None,jobs_dir=jobs)
        summary=collect(args,None,['btb-example'],None,result,'test-job')
        self.assertEqual(summary['runs'],[])
        self.assertEqual(summary['issues'],[])
        self.assertEqual(len(summary['pending']),3)
        self.assertFalse(list(out.rglob('verdicts/*.json')))

    def test_quota_bad_request_is_retried_but_invalid_inputs_are_not(self):
        runtime = ROOT / 'verifier-eval/verifiers/aaaj/.venv/bin/python'
        if not runtime.exists(): self.skipTest('Requires the native AAAJ dependency environment')
        source = ROOT / 'verifier-eval/verifiers/aaaj/agent_as_a_judge/llm/provider.py'
        code = '''
import importlib.util, sys
from litellm.exceptions import BadRequestError
spec=importlib.util.spec_from_file_location('quota_provider',sys.argv[1])
provider=importlib.util.module_from_spec(spec);spec.loader.exec_module(provider)
quota=BadRequestError(message='{"code": 429, "status": "RESOURCE_EXHAUSTED"}',model='gemini-3-flash',llm_provider='gemini')
invalid=BadRequestError(message='{"code": 400, "status": "INVALID_ARGUMENT"}',model='gemini-3-flash',llm_provider='gemini')
assert provider._retryable_provider_error(quota)
assert not provider._retryable_provider_error(invalid)
'''
        subprocess.run([str(runtime), '-c', code, str(source)], check=True, capture_output=True)


if __name__ == "__main__":
    unittest.main()

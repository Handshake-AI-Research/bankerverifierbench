"""Audit native Codex sessions without treating inaccessible evidence as defects."""
import json
import hashlib
import re
from pathlib import Path
from ..verdict import TokenUsage

# Standard API token estimates; preserve usage so prices can be recomputed.
# https://developers.openai.com/api/docs/pricing (retrieved 2026-10-06)
PRICES={'gpt-5.4':(2.5,.25,15),'gpt-5.4-mini':(.75,.075,4.5),
 'gpt-5.4-nano':(.2,.02,1.25),'gpt-6-sol':(2,.2,10),
 'gpt-6-luna':(.1,.01,.5),'gpt-6.1-sol':(2,.1,10)}


def deliverable_evidence_names(workspace: Path):
    """Include byte-identical workspace aliases of the actual deliverables."""
    def digest(path):
        with path.open('rb') as stream:
            return hashlib.file_digest(stream, 'sha256').digest()
    deliverables = [p for p in (workspace/'banker_workspace/deliverables').rglob('*') if p.is_file()]
    hashes = {digest(p) for p in deliverables}
    sizes = {p.stat().st_size for p in deliverables}
    return sorted({p.name for p in workspace.rglob('*')
                   if p.is_file() and p.stat().st_size in sizes and digest(p) in hashes})


def _mcp_failed(value):
    # MCP can transport a successful tool invocation containing an application
    # failure. Check both layers, including JSON inside text content blocks.
    if isinstance(value, str):
        try: return _mcp_failed(json.loads(value))
        except (ValueError, TypeError): return False
    if isinstance(value, list): return any(_mcp_failed(x) for x in value)
    if isinstance(value, dict):
        if value.get('isError') is True or value.get('success') is False: return True
        return any(_mcp_failed(x) for x in value.values() if isinstance(x, (str, list, dict)))
    return False


def _code_mode_results(value):
    """Read returned JSON envelopes, never execute the agent's JavaScript."""
    if isinstance(value, str):
        try:
            yield from _code_mode_results(json.loads(value))
        except (ValueError, TypeError):
            # Code-mode output can concatenate independently emitted JSON items.
            for line in value.splitlines():
                try:
                    yield from _code_mode_results(json.loads(line))
                except (ValueError, TypeError):
                    pass
    elif isinstance(value, list):
        for item in value:
            yield from _code_mode_results(item)
    elif isinstance(value, dict):
        if 'exit_code' in value or 'content' in value or 'isError' in value:
            yield value
        elif value.get('type') in {'input_text', 'text'}:
            yield from _code_mode_results(value.get('text', ''))
        elif 'result' in value:
            yield from _code_mode_results(value['result'])
        elif value.get('status') == 'fulfilled':
            yield from _code_mode_results(value.get('value'))


def _code_mode_evidence(call, output, deliverable_names, mcp_tools):
    source = call.get('input', '')
    results = list(_code_mode_results(output))
    commands = [r for r in results if 'exit_code' in r]
    # Conservative batch-level evidence: an unrelated successful command must
    # not conceal a failing read in the same wrapper. Opaque outputs need review.
    readers = ('openpyxl', 'load_workbook', 'pptx', 'pdfplumber', 'pypdf',
               'pdftotext', 'zipfile', 'unzip', 'elementtree', 'libreoffice',
               'soffice', 'pdftoppm', 'image.open', 'read_excel')
    evidence = source + '\n' + '\n'.join(str(r.get('output', '')) for r in commands)
    files = int(bool(commands) and all(r['exit_code'] == 0 for r in commands)
                and 'tools.exec_command(' in source
                and any(n in evidence for n in deliverable_names)
                and any(reader in source.lower() for reader in readers))
    invoked = {name.rsplit('__', 1)[-1] for name in
               re.findall(r'tools\.(mcp__[A-Za-z0-9_]+)\s*\(', source)}
    mcp = sum(1 for r in results if 'content' in r and not _mcp_failed(r)) if invoked & mcp_tools else 0
    failures = [dict(name='exec/exec_command', output=str(r.get('output', ''))[:1000])
                for r in commands if r['exit_code'] not in {0, None}]
    if invoked & mcp_tools:
        failures.extend(dict(name='exec/MCP', output=str(r)[:1000])
                        for r in results if 'content' in r and _mcp_failed(r))
    fatal = []
    for r in commands:
        text = str(r.get('output', ''))
        if ('bwrap:' in text or 'MCP tool call requires approval' in text
            or ('/home/verifier/workspace/' in text and 'permission denied' in text.lower())):
            fatal.append(text[:500])
    return files, mcp, failures, fatal


def audit_sessions(trace:Path,model:str,effort:str,deliverable_names:list[str],mcp_tools:set[str],
                   instructions:str='',criteria:list[str]|None=None,audit_name:str='codex-audit.json'):
    sessions=sorted((trace/'judge_trajectories/sessions').rglob('*.jsonl'))
    if not sessions:raise ValueError('No native Codex session trajectory captured')
    calls={};outputs={};usage=[];contexts=[];seen=set();fatal=[];user_text=[];windows=set();compactions=0
    for path in sessions:
        for line in path.read_text().splitlines():
            row=json.loads(line);p=row.get('payload',{})
            if row.get('type')=='compacted':compactions+=1
            if row.get('type')=='turn_context':contexts.append(p)
            if p.get('type')=='message' and p.get('role')=='user':
                user_text.extend(c.get('text','') for c in p.get('content',[]) if isinstance(c,dict))
            if p.get('type')=='function_call':calls[p['call_id']]=p
            if p.get('type')=='function_call_output':outputs[p['call_id']]=p.get('output','')
            if p.get('type')=='custom_tool_call':calls[p['call_id']]=p
            if p.get('type')=='custom_tool_call_output':outputs[p['call_id']]=p.get('output','')
            if p.get('type')=='token_count' and p.get('info'):
                info=p['info'];total=info.get('total_token_usage') or {};last=info.get('last_token_usage')
                if info.get('model_context_window'):windows.add(info['model_context_window'])
                key=(str(path),json.dumps(total,sort_keys=True))
                if last and key not in seen:seen.add(key);usage.append(last)
    if not contexts or any(c.get('model')!=model or c.get('effort')!=effort for c in contexts):
        raise ValueError('Codex model or reasoning effort differs from requested condition')
    actual_prompt='\n'.join(user_text)
    if instructions and instructions not in actual_prompt:
        raise ValueError('Frozen task prompt missing from actual native Codex user messages')
    if criteria and any(text not in actual_prompt for text in criteria):
        raise ValueError('Canonical rubric text missing from actual native Codex user messages')
    successful_files=successful_mcp=0;tool_failures=[]
    for ident,call in calls.items():
        raw_output=outputs.get(ident,'');output=str(raw_output);name=call.get('name','');args=str(call.get('arguments',''))
        if call.get('type') == 'custom_tool_call' and name == 'exec':
            files, mcp, failures, infrastructure = _code_mode_evidence(call, raw_output, deliverable_names, mcp_tools)
            successful_files += files
            successful_mcp += mcp
            tool_failures.extend(failures)
            fatal.extend(infrastructure)
            continue
        tool_name=name.rsplit('__',1)[-1].rsplit('.',1)[-1]
        if 'bwrap:' in output or 'MCP tool call requires approval' in output:fatal.append(output[:500])
        if ('/home/verifier/workspace/' in output and 'permission denied' in output.lower()):
            fatal.append('MCP-exported source evidence inaccessible: '+output[:500])
        failed=any(s in output.lower() for s in ('process exited with code 1','process exited with code 2',
                'permission denied','operation not permitted','requires approval','"iserror": true'))
        if tool_name in mcp_tools and _mcp_failed(raw_output): failed=True
        if failed:tool_failures.append(dict(name=name,output=output[:1000]))
        if name in {'exec_command','shell_command'} and 'Process exited with code 0' in output:
            evidence=args+'\n'+output
            if any(n in evidence for n in deliverable_names) and any(reader in evidence.lower() for reader in
                    ('openpyxl','load_workbook','pptx','pdfplumber','pypdf','pdftotext','sheet name:','worksheet',
                     'zipfile','unzip','elementtree','libreoffice','soffice','pdftoppm','image.open','read_excel')):
                successful_files+=1
        if tool_name in mcp_tools and output and not failed:successful_mcp+=1
    flags=[]
    metadata_fallback=any('Model metadata for' in p.read_text(errors='replace') and 'fallback metadata' in p.read_text(errors='replace')
                          for p in (trace/'judge_trajectories').glob('*stderr.txt'))
    if metadata_fallback:flags.append('Native Codex used fallback model metadata; effective context window recorded')
    if successful_mcp==0:flags.append('No successful independent BVB MCP source lookup in the judge trajectory')
    if tool_failures:flags.append(f'{len(tool_failures)} tool errors; inspect recovery and evidence coverage')
    prompt=sum(int(u.get('input_tokens',0)) for u in usage);cached=sum(int(u.get('cached_input_tokens',0)) for u in usage)
    completion=sum(int(u.get('output_tokens',0)) for u in usage)
    cache_writes=sum(int(u.get('cache_write_input_tokens',0)) for u in usage)
    reasoning=sum(int(u.get('reasoning_output_tokens',0)) for u in usage)
    rates=PRICES.get(model);cost=None
    if rates and usage:
        cost=0.0
        for u in usage:
            inp=int(u.get('input_tokens',0));cache=int(u.get('cached_input_tokens',0));out=int(u.get('output_tokens',0))
            writes=int(u.get('cache_write_input_tokens',0))
            if cache+writes>inp:raise ValueError('Native cache read/write tokens exceed total input tokens')
            multiplier=2 if inp>272000 and model not in {'gpt-5.4-mini','gpt-5.4-nano'} else 1
            # GPT-5.6+ cache writes replace the uncached rate with 1.25x;
            # this is not an additive fee. Reasoning is already in output.
            write_multiplier=1.25 if model.startswith('gpt-6') else 1
            cost+=((inp-cache-writes)*rates[0]*multiplier+cache*rates[1]*multiplier+
                   writes*rates[0]*write_multiplier*multiplier+out*rates[2]*(1.5 if multiplier==2 else 1))/1e6
    result=dict(sessions=len(sessions),model=model,reasoning_effort=effort,successful_deliverable_commands=successful_files,
        code_mode_wrapper_calls=sum(c.get('type')=='custom_tool_call' and c.get('name')=='exec' for c in calls.values()),
        native_prompt_verified=bool(instructions),native_criteria_verified=len(criteria or []),
        model_metadata_fallback=metadata_fallback,effective_context_windows=sorted(windows),
        largest_request_input_tokens=max((int(u.get('input_tokens',0)) for u in usage),default=0),
        native_compactions=compactions,
        successful_mcp_calls=successful_mcp,tool_failures=tool_failures,fatal_infrastructure_errors=fatal,flags=flags,
        model_cost_usd=cost,cost_basis='Native per-request token usage, Standard API rate estimate; infrastructure fees excluded',
        usage_requests=len(usage),tokens=TokenUsage(prompt_tokens=prompt,cached_prompt_tokens=cached,
            cache_write_prompt_tokens=cache_writes,completion_tokens=completion,reasoning_completion_tokens=reasoning).model_dump())
    (trace/audit_name).write_text(json.dumps(result,indent=2))
    if fatal or successful_files==0:raise ValueError('Codex could not inspect deliverables with its intended tools; run excluded')
    return result

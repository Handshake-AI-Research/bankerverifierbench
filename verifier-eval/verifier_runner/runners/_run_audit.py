"""Reject native error fallbacks instead of scoring them as genuine defects."""
import hashlib
import json
from pathlib import Path
import re
import subprocess


def reader_audit(trace: Path):
    """Keep document-extraction failures visible even when native grading succeeds."""
    warnings = []; reducto_metrics = []; local_metrics = []
    for name in ('worker-stdout.txt', 'worker-stderr.txt'):
        path = trace / name; proc = None
        if path.exists(): stream = path.open(errors='replace')
        elif Path(str(path) + '.zst').exists():
            proc = subprocess.Popen(['zstd', '-q', '-d', '-c', str(path) + '.zst'],
                                    stdout=subprocess.PIPE, text=True, errors='replace')
            stream = proc.stdout
        else: continue
        with stream:
            for line in stream:
                if re.search(r'Local extraction failed for |Reducto extraction failed for |'
                             r'Failed to extract (?:content|sub-artifact)|'
                             r'^Error reading|^No reader found|Failed to convert URL to base64:', line):
                    warning = line.strip()[:1500]
                    if warning not in warnings: warnings.append(warning)
                for label, target in (('Reducto API calls', reducto_metrics), ('Local extractions', local_metrics)):
                    match = re.search(re.escape(label) + r': (\d+) \(success: (\d+), failed: (\d+)\)', line)
                    if match: target.append(dict(total=int(match[1]), success=int(match[2]), failed=int(match[3])))
        if proc and proc.wait(): raise ValueError('Could not read archived native reader log')
    return dict(reader_warnings=warnings, reducto_metrics=reducto_metrics,
                local_extraction_metrics=local_metrics,
                requires_evidence_review=bool(warnings or any(x['failed'] for x in reducto_metrics + local_metrics)),
                note='A reader warning requires review; malformed reviewed artifacts are preserved, not repaired or silently removed.')


def criterion_errors(criteria):
    errors = []
    for criterion in criteria:
        reason = criterion.get('reason', '') if isinstance(criterion, dict) else criterion.reason
        index = criterion.get('index') if isinstance(criterion, dict) else criterion.index
        if re.match(r'^(?:default:\s*)?(?:Error parsing judge response:|Error calling (?:judge|LLM)|Grading failed:)', reason):
            errors.append(dict(index=index, reason=reason[:1500]))
    return errors


def call_audit(trace: Path):
    source = trace / 'llm_calls.jsonl'
    proc = None
    if source.exists():
        stream = source.open()
    else:
        proc = subprocess.Popen(['zstd', '-q', '-d', '-c', str(source) + '.zst'], stdout=subprocess.PIPE, text=True)
        stream = proc.stdout
    states = {}; failed = calls = truncated = 0; models = set(); cost = 0.; cost_complete = True
    inputs_path=trace/'judge-inputs.json'
    inputs=json.loads(inputs_path.read_text()) if inputs_path.exists() else {}
    expected=[c['criterion'] for c in inputs.get('rubric',[])]
    seen_criteria=set(); prompt_seen=False
    with stream:
        for line in stream:
            if not line.strip(): continue
            record = json.loads(line); calls += 1; models.add(record.get('model'))
            message_text='\n'.join(m.get('content','') if isinstance(m.get('content'),str) else
                                   '\n'.join(c.get('text','') for c in (m.get('content') or []) if isinstance(c,dict))
                                   for m in record.get('messages',[]))
            if inputs.get('instructions') and inputs['instructions'] in message_text:prompt_seen=True
            index=(record.get('metadata') or {}).get('criterion_index')
            if isinstance(index,int) and 0<=index<len(expected) and expected[index] in message_text:seen_criteria.add(index)
            signature = hashlib.sha256(json.dumps([record.get('model'), record.get('messages')], sort_keys=True).encode()).hexdigest()
            if record.get('error'):
                failed += 1; states[signature] = False
            elif record.get('response'):
                states[signature] = True
                if record.get('cost_usd') is None: cost_complete = False
                cost += record.get('cost_usd') or 0
                truncated += sum(c.get('finish_reason') == 'length' for c in record['response'].get('choices', []))
    if proc and proc.wait(): raise ValueError('Could not read archived LLM trajectory')
    return dict(recorded_calls=calls, models=sorted(models), failed_calls=failed,
                frozen_prompt_in_actual_messages=prompt_seen,
                expected_criteria=len(expected),criteria_seen_in_actual_messages=len(seen_criteria),
                criteria_missing_from_actual_messages=sorted(set(range(len(expected)))-seen_criteria),
                request_signatures_without_subsequent_success=sum(not x for x in states.values()),
                length_truncated_responses=truncated, recorded_model_cost_usd=cost,
                cost_complete=cost_complete)

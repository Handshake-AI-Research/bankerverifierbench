"""Explicit transport compatibility for the separately pinned model ablations."""
import importlib.metadata
import json
import os
from pathlib import Path


def configure():
    expected=os.getenv('BVB_LITELLM_VERSION')
    if not expected: return
    import litellm
    actual=importlib.metadata.version('litellm')
    if actual!=expected: raise RuntimeError(f'Expected LiteLLM {expected}, found {actual}')
    # GPT-6 reasoning rejects temperature=0. LiteLLM knows which provider
    # parameters are unsupported; retain the native reasoning configuration.
    litellm.drop_params=True
    trace=os.getenv('VERIFIER_TRACE_LLM_CALLS')
    if trace:
        path=Path(trace).parent/'transport-runtime.json'
        path.write_text(json.dumps({'litellm_version':actual,'drop_unsupported_parameters':True,
            'reasoning_effort':'native verifier setting / provider default',
            'temperature_policy':'native request; omitted by LiteLLM if unsupported'},indent=2))

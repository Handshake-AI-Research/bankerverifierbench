from pathlib import Path
import json
from bvb_contract import tree_hash,file_hash

def observe_clone(judge_input,trace_path):
    contract=json.loads(Path('/tests/input-contract.json').read_text())['identity']
    clone=Path(judge_input.workdir)
    payload=judge_input.model_dump()
    hashes=dict(workspace_sha256=tree_hash(clone/'workspace'),trajectory_sha256=file_hash(clone/'trajectory.json'))
    if any(hashes[k]!=contract[k] for k in hashes):raise ValueError('Private judge copy differs from reviewed evidence')
    Path(trace_path+'.input.json').write_text(json.dumps(dict(input=payload,clone_hashes=hashes,canonical_identity=contract),indent=2))

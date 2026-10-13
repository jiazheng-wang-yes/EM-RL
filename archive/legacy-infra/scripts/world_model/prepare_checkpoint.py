"""Fetch the exact released LeWM PushT checkpoint used by the screens."""
import hashlib
from pathlib import Path

from huggingface_hub import snapshot_download

WORK=Path(__file__).resolve().parents[2]
DEST=WORK/'checkpoints/quentinll/lewm-pusht'
EXPECTED='48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607'
REVISION='22b330c28c27ead4bfd1888615af1340e3fe9052'

snapshot_download('quentinll/lewm-pusht',revision=REVISION,
                  allow_patterns=['config.json','weights.pt'],local_dir=DEST)
observed=hashlib.sha256((DEST/'weights.pt').read_bytes()).hexdigest()
if observed!=EXPECTED:
    raise RuntimeError(f'Unexpected checkpoint hash: {observed}')
print(f'Checkpoint ready at {DEST}; SHA-256 {observed}')

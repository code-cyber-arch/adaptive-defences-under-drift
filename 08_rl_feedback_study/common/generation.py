"""Reproducible stream and intervention generation parameters."""
import hashlib
import json
BLOCK = 1000
BURST = 250
WARMUP = 1000

def stable_seed(*parts):
    """Keep random generation reproducible for the same identifiers."""
    text = json.dumps(parts, separators=(',', ':'), ensure_ascii=True)
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], 'big')

"""Load public observations separately from protected and evaluator evidence."""
import numpy as np
import pandas as pd
from common import protocol as p

def protected(root, condition):
    """Read the reserved validation sample and its release times."""
    return pd.read_parquet(root / '01_attacks' / condition['audit'])

def observations(root, condition):
    """Attach row identity and reservation flags without loading hidden truth."""
    frame = pd.read_parquet(root / '01_attacks' / condition['observations'])
    if 'row_id' not in frame:
        frame.insert(0, 'row_id', np.arange(len(frame)))
    reserved = np.zeros(len(frame), dtype=bool)
    ids = protected(root, condition).row_id.to_numpy(dtype=int)
    reserved[ids] = True
    frame['audit_reserved'] = reserved
    return frame

def truth(root, condition):
    """Load scoring references for offline analysis only."""
    frame = pd.read_parquet(root / '01_attacks' / condition['truth'])
    reserved = np.zeros(len(frame), dtype=bool)
    reserved[protected(root, condition).row_id.to_numpy(dtype=int)] = True
    frame['audit_reserved'] = reserved
    return frame

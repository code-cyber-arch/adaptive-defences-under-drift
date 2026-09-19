"""Content checks shared by builders and the verifier."""
import numpy as np
ANNOTATIONS = {'target', 'target-class', 'target-class-name', 'label', 'class', '__label'}

def annotation(name):
    """Identify names that may expose target annotations."""
    name = str(name).lower().split('::')[-1]
    return name in ANNOTATIONS or name.startswith(('target-', 'target_', 'label_', '__'))

def frame_check(frame):
    """Check the numeric feature and label schema."""
    expected = [f'f{i}' for i in range(len(frame.columns) - 1)] + ['label']
    if list(frame.columns) != expected:
        raise ValueError('Expected numeric f0..fN + label, without metadata columns')
    if not np.isfinite(frame.iloc[:, :-1].to_numpy(dtype=float)).all():
        raise ValueError('Nonfinite feature value')
    labels = frame.label.to_numpy()
    if not np.isfinite(labels).all() or not np.equal(labels, labels.astype(int)).all() or (labels < 0).any():
        raise ValueError('Labels must be nonnegative integers')

def lineage_check(mapping):
    """Reject target annotations from the feature lineage."""
    if not mapping or any((annotation(v['source_column']) for v in mapping.values())):
        raise ValueError('Feature lineage contains target/annotation metadata')

def exposed_blocks(intervals, n_rows, block):
    """Exact union of blocks intersecting half-open row intervals."""
    result = np.zeros((n_rows + block - 1) // block, dtype=bool)
    for start, end in intervals:
        if not 0 <= start < end <= n_rows:
            raise ValueError('Interval outside stream')
        result[start // block:(end - 1) // block + 1] = True
    return result

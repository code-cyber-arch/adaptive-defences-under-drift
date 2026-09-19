"""Detector adapters with matching feature and feedback interfaces."""
from common import protocol as p
monitors = p.load_module('comparison_monitors', '02_detectors/monitors.py')

def create(policy, settings):
    """Construct the requested drift detector."""
    name = settings.get('name', 'adwin').lower() if policy != 'none' else 'none'
    return monitors.make(name, seed=settings.get('seed', 0))

def update(detector, errors, features=None):
    """Feed the available observations to the detector and return its alarm."""
    import numpy as np
    if features is None:
        features = np.zeros((len(errors), 1))
    return detector.update(features, errors)

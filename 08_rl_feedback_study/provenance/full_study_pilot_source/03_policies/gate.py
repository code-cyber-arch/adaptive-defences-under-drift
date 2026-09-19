"""Response timing and validation of candidate updates."""
from collections import deque
import numpy as np

class Response:

    def __init__(self, name, history=5, wait=2, margin=0.02):
        """Initialise this component with its declared settings and empty state."""
        if name not in ['none', 'immediate', 'confirmed']:
            raise ValueError(name)
        self.name = name
        self.history = deque(maxlen=history)
        self.wait = wait
        self.margin = margin
        self.pending = None
        self.baseline = None
        self.after = []

    def propose(self, block, error, fired):
        """Apply the fixed response timing rule to the latest block."""
        requested = None
        event = 'none'
        if self.name == 'immediate' and fired:
            requested = [block]
            event = 'reset_proposed'
        elif self.name == 'confirmed':
            if self.pending is not None:
                self.after.append(error)
                if block - self.pending >= self.wait:
                    if np.mean(self.after) > self.baseline + self.margin:
                        requested = list(range(self.pending, block + 1))
                        event = 'reset_proposed'
                    else:
                        event = 'candidate_discarded'
                    self.pending = None
                    self.after = []
            elif fired:
                if len(self.history) == self.history.maxlen:
                    self.pending = block
                    self.baseline = float(np.mean(self.history))
                    self.after = []
                    event = 'candidate_opened'
                else:
                    event = 'insufficient_history'
        self.history.append(error)
        return (requested, event)

    def committed(self):
        """Clear pending response state after an accepted reset."""
        self.history.clear()
        self.pending = None
        self.after = []

def assess(active, candidate, features, labels, classes, settings):
    """Accept a candidate only when protected-sample accuracy does not decrease."""
    result = {'accepted': False, 'reason': 'no_protected_sample', 'audit_n': len(labels), 'active_accuracy': None, 'candidate_accuracy': None, 'accuracy_change': None}
    if len(labels) == 0:
        return result
    accuracy = []
    for model in [active, candidate]:
        predictions = []
        for x in features:
            probabilities = model.predict_proba_one(x)
            predictions.append(max(probabilities, key=probabilities.get) if probabilities else -1)
        accuracy.append(float(np.mean(np.asarray(predictions) == labels)))
    accepted = bool(accuracy[1] >= accuracy[0])
    result.update(accepted=accepted, reason='accepted' if accepted else 'accuracy_decreased', active_accuracy=accuracy[0], candidate_accuracy=accuracy[1], accuracy_change=accuracy[1] - accuracy[0])
    return result

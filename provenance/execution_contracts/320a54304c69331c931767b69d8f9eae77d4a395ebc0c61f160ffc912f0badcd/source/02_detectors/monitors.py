"""Block interfaces for error and feature distribution monitors."""
import warnings
import numpy as np
from river import drift
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

class NullDetector:
    channel = 'none'
    variant = 'null'
    statistic = None

    def update(self, X, error):
        """Feed the available observations to the detector and return its alarm."""
        return False

    def reset(self):
        """Clear the detector state after a committed reset."""
        pass

class ErrorDetector:
    channel = 'error'
    statistic = None

    def __init__(self, name):
        """Initialise this component with its declared settings and empty state."""
        self.name = name
        self.variant = 'river_adwin' if name == 'adwin' else 'river_hddmw'
        self.reset()

    def reset(self):
        """Clear the detector state after a committed reset."""
        self.inner = drift.ADWIN(delta=0.002) if self.name == 'adwin' else drift.binary.HDDMW(drift_confidence=0.001, warning_confidence=0.005)

    def update(self, X, error):
        """Feed the available observations to the detector and return its alarm."""
        fired = False
        for e in error:
            self.inner.update(float(e) if self.name == 'adwin' else bool(e))
            fired = bool(self.inner.drift_detected) or fired
        return fired

class HellingerDetector:
    channel = 'feature'
    variant = 'fixed_reference_hellinger_difference'

    def __init__(self, gamma=1.5):
        """Initialise this component with its declared settings and empty state."""
        self.gamma = gamma
        self.reset()

    def reset(self):
        """Clear the detector state after a committed reset."""
        self.reference = None
        self.edges = None
        self.previous = None
        self.history = []
        self.statistic = None

    @staticmethod
    def distance(a, b):
        """Calculate the Hellinger distance between two histograms."""
        if a.sum() == 0 or b.sum() == 0:
            raise ValueError('Empty histogram')
        return float(np.sqrt(max(0, 1 - np.sqrt(a / a.sum() * b / b.sum()).sum())))

    def histograms(self, X):
        """Summarise each feature using the stored histogram bins."""
        return [np.histogram(X[:, j], bins=self.edges[j])[0].astype(float) for j in range(X.shape[1])]

    def update(self, X, error):
        """Feed the available observations to the detector and return its alarm."""
        if not np.isfinite(X).all():
            raise ValueError('Nonfinite detector input')
        if self.reference is None:
            bins = max(5, int(np.sqrt(len(X))))
            self.edges = [np.r_[-np.inf, np.histogram_bin_edges(X[:, j], bins=bins), np.inf] for j in range(X.shape[1])]
            self.reference = self.histograms(X)
            return False
        current = self.histograms(X)
        self.statistic = float(np.mean([self.distance(a, b) for a, b in zip(self.reference, current)]))
        if self.previous is None:
            self.previous = self.statistic
            return False
        change = abs(self.statistic - self.previous)
        self.previous = self.statistic
        fired = len(self.history) >= 3 and change > np.mean(self.history) + self.gamma * np.std(self.history)
        self.history.append(change)
        if fired:
            self.reference = current
            self.history = []
            self.previous = None
        return bool(fired)

class D3Detector:
    channel = 'feature'
    variant = 'out_of_fold_auc_difference'

    def __init__(self, seed=0, w=5, recent=2, gamma=2.0, tau=None):
        """Initialise this component with its declared settings and empty state."""
        self.seed, self.w, self.recent, self.gamma, self.tau = (seed, w, recent, gamma, tau)
        if w < 1 or recent < 1:
            raise ValueError('Positive window sizes required')
        self.reset()

    def reset(self):
        """Clear the detector state after a committed reset."""
        self.old, self.new, self.history = ([], [], [])
        self.previous = None
        self.statistic = None

    def auc(self, A, B):
        """Measure old-versus-recent separation using out-of-fold scores."""
        X = np.vstack([A, B])
        y = np.r_[np.zeros(len(A)), np.ones(len(B))]
        pred = np.empty(len(y), dtype=float)
        folds = StratifiedKFold(n_splits=2, shuffle=True, random_state=self.seed)
        for train, test in folds.split(X, y):
            scaler = StandardScaler().fit(X[train])
            classifier = LogisticRegression(max_iter=500, solver='liblinear', random_state=self.seed)
            with warnings.catch_warnings():
                warnings.simplefilter('error', ConvergenceWarning)
                classifier.fit(scaler.transform(X[train]), y[train])
            pred[test] = classifier.decision_function(scaler.transform(X[test]))
        return float(roc_auc_score(y, pred))

    def update(self, X, error):
        """Feed the available observations to the detector and return its alarm."""
        if not np.isfinite(X).all():
            raise ValueError('Nonfinite detector input')
        if len(self.old) < self.w:
            self.old.append(X.copy())
            return False
        self.new.append(X.copy())
        if len(self.new) < self.recent:
            return False
        self.statistic = self.auc(np.vstack(self.old), np.vstack(self.new))
        if self.tau is not None:
            fired = self.statistic > self.tau
        elif self.previous is None:
            fired = False
        else:
            change = abs(self.statistic - self.previous)
            fired = len(self.history) >= 3 and change > np.mean(self.history) + self.gamma * np.std(self.history)
            self.history = (self.history + [change])[-50:]
        self.previous = self.statistic
        if fired:
            self.old, self.new, self.history, self.previous = (list(self.new), [], [], None)
        else:
            self.old, self.new = ((self.old + self.new)[-self.w:], [])
        return bool(fired)

def make(name, seed=0):
    """Select the requested detector implementation."""
    if name == 'none':
        return NullDetector()
    if name in ['adwin', 'hddm_w']:
        return ErrorDetector(name)
    if name == 'hellinger':
        return HellingerDetector()
    if name == 'd3_oof':
        return D3Detector(seed=seed)
    if name == 'd3_fixed':
        d = D3Detector(seed=seed, tau=0.7)
        d.variant = 'out_of_fold_auc_fixed_threshold'
        return d
    raise ValueError(f'Unknown detector: {name}')

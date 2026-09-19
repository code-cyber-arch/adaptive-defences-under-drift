"""Small Q-tables using alarms, optional error history and delayed accuracy."""
from collections import deque
from copy import deepcopy
import numpy as np
ACTIONS = ('update', 'reset')

def accuracy(model, xs, ys):
    """Calculate the model accuracy on a protected sample."""
    predictions = []
    for x in xs:
        probabilities = model.predict_proba_one(x)
        predictions.append(max(probabilities, key=probabilities.get) if probabilities else -1)
    return float(np.mean(np.asarray(predictions) == ys))

class Controller:

    def __init__(self, settings, seed, training=False, table=None, collect_rewards=True):
        """Initialise this component with its declared settings and empty state."""
        self.settings = dict(settings)
        self.seed = seed
        self.training = training
        if training and not collect_rewards:
            raise ValueError('Training requires reward collection')
        self.collect_rewards = collect_rewards
        self.rng = np.random.default_rng(seed)
        self.variant = settings.get('variant', 'alarm')
        if self.variant not in ('alarm', 'alarm_persistence'):
            raise ValueError('Unknown RL state representation')
        self.persistence = settings.get('persistence', {'history': 5, 'recent': 2, 'margin': 0.02})
        if self.persistence['history'] < 1 or self.persistence['recent'] < 1 or self.persistence['margin'] < 0:
            raise ValueError('Invalid error-history settings')
        states = ('0', '1') if self.variant == 'alarm' else ('00', '01', '10', '11')
        self.table = deepcopy(table if table is not None else {s: [0.0, 0.0] for s in states})
        if set(self.table) != set(states) or any(len(v) != 2 for v in self.table.values()):
            raise ValueError('Q-table does not match the declared states and actions')
        if not all(np.isfinite(v).all() for v in self.table.values()):
            raise ValueError('Q-values must be finite')
        self.pending = None
        self.transitions = []
        self.visits = {}
        self.epsilon = settings['epsilon'] if training else 0.0
        self.begin_episode()

    def begin_episode(self):
        """Clear stream history without discarding the learned Q-values."""
        self.errors = deque(maxlen=self.persistence['history'] + self.persistence['recent'])
        self.pending = None
        self.evidence = {'rl_error_ready': False, 'rl_error_baseline': None,
                         'rl_error_recent': None, 'rl_error_elevated': False}

    def state(self, fired, error=None):
        """Use only completed pre-update block errors, never future or attack truth."""
        if error is not None:
            if not np.isfinite(error) or not 0 <= error <= 1:
                raise ValueError('Block error must be a finite proportion')
            self.errors.append(float(error))
        ready = len(self.errors) == self.errors.maxlen
        baseline = recent = None
        elevated = False
        if ready:
            values = list(self.errors)
            baseline = float(np.mean(values[:self.persistence['history']]))
            recent = float(np.mean(values[self.persistence['history']:]))
            elevated = recent > baseline + self.persistence['margin']
        self.evidence = {'rl_error_ready': ready, 'rl_error_baseline': baseline,
                         'rl_error_recent': recent, 'rl_error_elevated': elevated}
        alarm = str(int(bool(fired)))
        return alarm if self.variant == 'alarm' else alarm + str(int(elevated))

    def values(self, state):
        """Read the two action values for an alarm state."""
        return np.asarray(self.table[state], dtype=float)

    def choose(self, state):
        """Choose continue learning or reset; explore only during training."""
        if self.training and self.rng.random() < self.epsilon:
            index = int(self.rng.integers(2))
        else:
            index = int(np.argmax(self.values(state)))
        self.visits[state] = self.visits.get(state, 0) + 1
        return ACTIONS[index]

    def record(self, state, action, after, block, work, rejected):
        """Keep the chosen action and resulting model until its reward is released."""
        if not self.collect_rewards:
            return
        if action not in ACTIONS:
            raise ValueError('Unknown action')
        self.pending = (state, action, after, block, work, rejected)

    def settle(self, audits, features, block_size, now, next_state, terminal=False):
        """Use the newly released protected sample to calculate delayed reward."""
        if self.pending is None:
            return None
        state, action, model, source_block, work, rejected = self.pending
        if now <= source_block:
            raise AssertionError('Reward is not delayed')
        sample = audits[(audits.row_id // block_size == source_block) & (audits.release_block <= now)]
        if len(sample) == 0:
            raise ValueError('No released protected reward sample')
        assert int(sample.row_id.max()) < now * block_size
        reward = accuracy(model, sample[features].to_dict('records'), sample.label.to_numpy())
        values = self.values(state)
        index = ACTIONS.index(action)
        old = float(values[index])
        # Frozen evaluation records rewards but never changes the Q-table.
        if self.training:
            target = reward + (0.0 if terminal else self.settings['discount'] * float(self.values(next_state).max()))
            values[index] += self.settings['learning_rate'] * (target - values[index])
            self.table[state] = values.tolist()
        self.transitions.append({'action_block': source_block, 'reward_block': now, 'state': state, 'action': action, 'reward': reward, 'audit_n': len(sample), 'audit_latest_row': int(sample.row_id.max()), 'work': work, 'rejected': rejected, 'q_before': old, 'q_after': float(values[index]), 'training': self.training, 'terminal': terminal})
        self.pending = None
        return reward

    def artifact(self):
        """Package the learned Q-table and its training settings."""
        return {'algorithm': 'tabular_q_learning', 'variant': self.variant, 'seed': self.seed, 'settings': self.settings, 'table': self.table, 'state_visits': self.visits, 'training_transitions': len(self.transitions), 'state_count': len(self.table), 'actions': list(ACTIONS)}

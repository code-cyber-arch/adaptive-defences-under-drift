"""Block execution with candidate isolation and delayed protected audits."""
from copy import deepcopy
import json
import time
import numpy as np
import pandas as pd
from river import tree
from common import protocol as p
gate_module = p.load_module('update_gate', '03_policies/gate.py')
detectors = p.load_module('stream_detectors', '02_detectors/detectors.py')
Response, assess = (gate_module.Response, gate_module.assess)

def execute(observations, audits, classes, policy, guarded, config, controller=None):
    """Predict each block, screen its proposed update and record the result."""
    feature_names = [c for c in observations if c.startswith('f') and c[1:].isdigit()]
    if set(observations) - set(feature_names) - {'row_id', 'label', 'audit_reserved'}:
        raise ValueError('Observation channel contains unexpected metadata')
    if set(audits) - set(feature_names) - {'row_id', 'label', 'release_block'}:
        raise ValueError('Audit channel contains unexpected metadata')
    if not observations.row_id.eq(np.arange(len(observations))).all():
        raise ValueError('Observation row identities are not contiguous')
    n = len(observations)
    block_size = config['block_size']
    warmup = config['warmup_rows']
    reserved = observations.audit_reserved.to_numpy(dtype=bool)
    if set(audits.row_id) != set(np.flatnonzero(reserved)):
        raise ValueError('Audit reservation differs from protected channel')
    model = tree.HoeffdingTreeClassifier()
    detector = detectors.create(policy, config['detector'])
    response = Response('none' if p.is_rl(policy) else policy, **config['response'])
    if p.is_rl(policy) and (controller is None or not guarded or config['audit']['delay_blocks'] != 1):
        raise ValueError('RL requires a controller, screening and one-block audit delay')
    if controller is not None:
        controller.begin_episode()
    prediction = np.full(n, -1, dtype=np.int64)
    probabilities = np.full((n, len(classes)), np.nan)
    admitted = np.full(n, -1, dtype=np.int64)
    attempt_visits = np.zeros(n, dtype=np.int64)
    accepted_visits = np.zeros(n, dtype=np.int64)
    events = []
    values = observations[feature_names].to_numpy(dtype=float)
    labels = observations.label.to_numpy(dtype=int)
    blocks = {}
    train_total = accepted_total = refit_total = actual_resets = 0
    clock_start = time.perf_counter()
    for block, left in enumerate(range(0, n, block_size)):
        right = min(left + block_size, n)
        row_ids = np.arange(left, right)[~reserved[left:right]]
        features = [dict(zip(feature_names, values[i])) for i in row_ids]
        blocks[block] = (row_ids, features, labels[row_ids])
        for old in list(blocks):
            if old < block - config['response']['wait']:
                del blocks[old]
        tick = time.perf_counter()
        # Predict before this block is used to fit a candidate.
        for i, x in zip(row_ids, features):
            probs = model.predict_proba_one(x)
            prediction[i] = int(max(probs, key=probs.get)) if probs else -1
            probabilities[i] = [float(probs.get(c, 0)) for c in classes]
        predict_seconds = time.perf_counter() - tick
        error = float(np.mean(prediction[row_ids] != labels[row_ids])) if len(row_ids) else 0.0
        fired = False
        requested = None
        response_event = 'warmup'
        if left >= warmup:
            fired = detectors.update(detector, prediction[row_ids] != labels[row_ids], values[row_ids])
            requested, response_event = response.propose(block, error, fired)
        earliest = block - config['audit']['delay_blocks'] - config['audit']['window_blocks'] + 1
        available = audits[(audits.release_block <= block) & (audits.row_id // block_size >= earliest)]
        rl_action = ''
        rl_state = ''
        delivered_reward = None
        if controller is not None and left >= warmup:
            rl_state = controller.state(fired, error)
            delivered_reward = controller.settle(audits, feature_names, block_size, block, rl_state)
            rl_action = controller.choose(rl_state)
            requested = [block] if rl_action == 'reset' else None
            response_event = 'rl_' + rl_action
        proposed_blocks = requested if requested else [block]
        train_ids = np.concatenate([blocks[b][0] for b in proposed_blocks]) if proposed_blocks else np.array([], dtype=int)
        if np.any(reserved[train_ids]) or np.any(train_ids >= right):
            raise AssertionError('Audit or future data reached training')
        tick = time.perf_counter()
        candidate = tree.HoeffdingTreeClassifier() if requested else deepcopy(model) if guarded and left >= warmup else model
        for b in proposed_blocks:
            _, xs, ys = blocks[b]
            for x, y in zip(xs, ys):
                candidate.learn_one(x, int(y))
        candidate_seconds = time.perf_counter() - tick
        attempt_visits[train_ids] += 1
        train_total += len(train_ids)
        gate = {'accepted': True, 'reason': 'warmup' if left < warmup else 'unscreened', 'audit_n': 0, 'active_accuracy': None, 'candidate_accuracy': None, 'accuracy_change': None}
        latest_audit = -1
        tick = time.perf_counter()
        if guarded and left >= warmup:
            earliest = block - config['audit']['delay_blocks'] - config['audit']['window_blocks'] + 1
            available = audits[(audits.release_block <= block) & (audits.row_id // block_size >= earliest)]
            audit_x = available[feature_names].to_dict('records')
            audit_y = available.label.to_numpy(dtype=int)
            latest_audit = int(available.row_id.max()) if len(available) else -1
            if latest_audit >= left:
                raise AssertionError('Audit channel contains current or future observations')
            gate = assess(model, candidate, audit_x, audit_y, classes, config['gate'])
        gate_seconds = time.perf_counter() - tick
        accepted = gate['accepted']
        # Only an accepted candidate can replace the active learner.
        if accepted:
            model = candidate
            accepted_visits[train_ids] += 1
            newly = train_ids[admitted[train_ids] < 0]
            admitted[newly] = block
            accepted_total += len(train_ids)
            if requested:
                actual_resets += 1
                refit_total += len(train_ids)
                response.committed()
                detector = detectors.create(policy, config['detector'])
        if controller is not None and left >= warmup:
            controller.record(rl_state, rl_action, deepcopy(model), block, len(train_ids), not accepted)
        events.append({'rl_action': rl_action, 'rl_state': rl_state, 'rl_reward_delivered': delivered_reward, 'block': block, 'start_row': left, 'end_row': right, 'scored': left >= warmup, 'feedback_error': error, 'raw_fire': fired, 'response_event': response_event, 'reset_requested': bool(requested), 'reset_committed': bool(requested and accepted), 'training_blocks': json.dumps(proposed_blocks), 'candidate_rows': len(train_ids), 'accepted': accepted, 'audit_latest_row': latest_audit, 'candidate_seconds': candidate_seconds, 'predict_seconds': predict_seconds, 'gate_seconds': gate_seconds, **{k: v for k, v in gate.items() if k != 'supported_classes'}})
        if controller is not None:
            events[-1].update(controller.evidence)
    if controller is not None and controller.pending is not None:
        controller.settle(audits, feature_names, block_size, block + 1, 'terminal', terminal=True)
    result = pd.DataFrame({'row_id': np.arange(n), 'prediction': prediction, 'audit_reserved': reserved, 'scored': (np.arange(n) >= warmup) & ~reserved, 'first_admitted_block': admitted, 'candidate_training_visits': attempt_visits, 'accepted_training_visits': accepted_visits})
    for j, c in enumerate(classes):
        result[f'p_{c}'] = probabilities[:, j]
    if (admitted[reserved] >= 0).any():
        raise AssertionError('Protected audit data entered the learner')
    stats = {'rows': n, 'training_visits': train_total, 'accepted_training_visits': accepted_total, 'accepted_refit_rows': refit_total, 'resets': actual_resets, 'seconds': time.perf_counter() - clock_start, 'pending_candidate': response.pending is not None, 'audit_rows_reserved': int(reserved.sum()), 'policy': policy, 'guarded': guarded}
    return (result, pd.DataFrame(events), stats)

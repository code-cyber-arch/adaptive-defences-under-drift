"""Strict reuse of records whose per-run scientific settings are unchanged."""
from copy import deepcopy
from common import protocol as p


def effective_config(config, training=False):
    result = deepcopy(config)
    result['rl'].pop('seeds', None)
    if not training:
        result.pop('base_seeds', None)
        result.pop('radar_attack_assignments', None)
    return result


def training_request_matches(digest, source):
    current = source / 'request.json'
    if digest == p.sha(current):
        return True
    archived = p.ROOT / 'provenance/requests' / f'{digest}.json'
    if not archived.exists() or p.sha(archived) != digest:
        return False
    return effective_config(p.read(archived)['config'], True) == effective_config(p.read(current)['config'], True)


def run_contract_matches(saved, requested):
    old, new = deepcopy(saved), deepcopy(requested)
    old['config'] = effective_config(old['config'])
    new['config'] = effective_config(new['config'])
    return old == new


def counts(spec):
    datasets, detectors, seeds = len(spec['datasets']), len(spec['detectors']), len(spec['rl_seeds'])
    conditions = 14 * len(spec['evaluation_seeds']) + 1 + 6 * len(spec['radar']['evaluation_attack_assignments'])
    feedback_conditions = 4 * len(spec['evaluation_seeds']) + 1 + len(spec['radar']['evaluation_attack_assignments'])
    defaults = feedback_conditions * detectors * (4 + seeds)
    sensitivity = defaults * len(spec['feedback_fractions']) * len(spec['feedback_delays'])
    validation = datasets * detectors * 7 * 16
    evaluation = conditions * detectors * (4 + 2 * seeds)
    return dict(controllers=datasets*detectors*2*seeds, validation=validation, evaluation=evaluation,
                sensitivity=sensitivity, shared_defaults=defaults,
                additional_feedback=sensitivity-defaults, executions=validation+evaluation+sensitivity-defaults)

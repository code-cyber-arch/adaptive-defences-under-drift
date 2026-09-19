"""Paths, contracts and portable experiment records."""
from pathlib import Path
import datetime
import hashlib
import importlib.metadata
import json
import platform
ROOT = Path(__file__).resolve().parents[1]
VERSION = 'guarded-update-rl-3.0'
POLICIES = ['none', 'immediate', 'confirmed']
RL_POLICIES = ['rl_alarm', 'rl']


def is_rl(policy):
    """Identify either learned response policy."""
    return policy in RL_POLICIES

def sha(path):
    """Calculate the SHA-256 fingerprint of a file."""
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()

def seed(*parts):
    """Derive a reproducible seed from the supplied identifiers."""
    return int.from_bytes(hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).digest()[:4], 'big')

def read(path):
    """Read a UTF-8 JSON record."""
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write(path, value):
    """Write a complete JSON record atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)

def now():
    """Return the current UTC timestamp."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def sources():
    """Record hashes of code and dependency specifications."""
    paths = sorted((ROOT / 'common').glob('*.py')) + sorted((ROOT / 'tools').glob('*.py'))
    for folder in sorted(ROOT.glob('[0-9][0-9]_*')):
        paths.extend(sorted(folder.glob('*.py')))
    paths.extend([ROOT / 'requirements.txt', ROOT / 'requirements.lock.txt'])
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}

def arms(config):
    """List the unique comparison arms without duplicating no-reset references."""
    result = [{'policy': 'none', 'guarded': g, 'detector': 'none', 'rl_seed': None} for g in [False, True]]
    for detector in config.get('detectors', ['adwin']):
        for policy in ['immediate', 'confirmed']:
            result.extend(({'policy': policy, 'guarded': g, 'detector': detector, 'rl_seed': None} for g in [False, True]))
    for model in config.get('rl_models', []):
        variant = model.get('variant', 'alarm')
        if variant not in ('alarm', 'alarm_persistence'):
            raise ValueError('Unknown frozen RL variant')
        policy = 'rl_alarm' if variant == 'alarm' else 'rl'
        result.append({'policy': policy, 'guarded': True, 'detector': 'adwin', 'rl_seed': model['seed'], 'rl_model': model})
    return result

def run_id(condition, arm):
    """Create a stable identifier for a condition and policy arm."""
    suffix = f"__r{arm['rl_seed']}" if arm.get('rl_seed') is not None else ''
    return f"{condition}__{arm['detector']}__{arm['policy']}__{('guarded' if arm['guarded'] else 'unguarded')}{suffix}"

def environment():
    """Record the Python and package versions used for execution."""
    return {'python': platform.python_version(), 'platform': platform.platform(), 'packages': {name: importlib.metadata.version(name) for name in ['river', 'numpy', 'pandas', 'pyarrow', 'scikit-learn', 'matplotlib']}}

def output(path):
    """Require outputs to stay inside this project results folder."""
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT / 'results'):
        raise ValueError('Outputs must remain under this project results directory')
    return path

def load_module(name, relative):
    """Load code from a numbered phase folder."""
    import importlib.util
    import sys
    phase_root = ROOT
    if str(phase_root) not in sys.path:
        sys.path.insert(0, str(phase_root))
    spec = importlib.util.spec_from_file_location(name, phase_root / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def run_folder(root, condition, arm):
    """Give every run a readable dataset/attack/seed/policy location."""
    dataset = {'SEA_A': 'SEA', 'RBF_I': 'RBF', 'radar': 'RADAR'}[condition['stream']]
    base = 'capture' if condition['base_seed'] is None else f"base_seed_{condition['base_seed']}"
    folder = root / '05_runs' / dataset / condition['mode'] / condition['level'] / base
    if condition.get('attack_seed') is not None:
        folder /= f"attack_seed_{condition['attack_seed']}"
    detector = 'no_detector' if arm['detector'] == 'none' else arm['detector']
    policy = {'none': 'no_reset', 'immediate': 'immediate_reset', 'confirmed': 'confirmed_reset', 'rl_alarm': 'rl_alarm', 'rl': 'rl_persistence'}[arm['policy']]
    folder = folder / detector / policy / ('screened' if arm['guarded'] else 'unscreened')
    if arm.get('rl_seed') is not None:
        folder /= f"rl_seed_{arm['rl_seed']}"
    return folder

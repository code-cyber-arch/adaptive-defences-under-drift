"""Verify detector-specific fitting, model identity and matched comparisons."""
from pathlib import Path
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import tempfile
import unittest
from unittest.mock import patch
import test_update_safety as safety
from common import protocol as p
training = p.load_module('tested_detector_training', '04_rl_training/train.py')
worker = p.load_module('tested_detector_worker', '05_experiment/policy_robustness.py')
analysis = p.load_module('tested_detector_analysis', '06_analysis/policy_robustness.py')
DETECTORS = ['adwin', 'hddm_w', 'hellinger', 'd3_oof']


class DetectorTrainingTests(unittest.TestCase):
    def test_each_detector_is_used_in_real_training_and_recorded(self):
        fixture = safety.EngineTests(); fixture.setUp()
        config = deepcopy(fixture.config)
        config.update(dataset='SEA_A', base_seeds=[112], detectors=DETECTORS,
                      rl=dict(p.read(p.ROOT / 'configs/thesis.json')['rl'], epochs=1))
        condition = {'base_seed': 112, 'stream': 'SEA_A', 'key': 'train_fixture', 'classes': [0, 1]}
        with tempfile.TemporaryDirectory(dir=p.ROOT / 'tests') as tmp:
            root = Path(tmp)
            p.write(root / 'request.json', {'config': config})
            p.write(root / '01_attacks/manifest.json', {'conditions': {'train_fixture': condition}, 'dataset':'SEA_A', 'data_role':'training'})
            for detector in DETECTORS:
                for variant in ['alarm', 'alarm_persistence']:
                    calls = []
                    original = training.engine.execute
                    def execute(*args):
                        calls.append(args[5]['detector']['name'])
                        return original(*args)
                    with patch.object(training.channels, 'observations', return_value=fixture.obs), \
                         patch.object(training.channels, 'protected', return_value=fixture.audit), \
                         patch.object(training.engine, 'execute', side_effect=execute):
                        model = training.train_one(root, config, 7, variant, detector)
                    self.assertEqual(calls, [detector])
                    saved = p.read(p.ROOT / model['path'])
                    self.assertEqual(saved['detector'], detector)
                    self.assertEqual(saved['variant'], variant)
                    self.assertEqual(saved['training_base_seeds'], [112])
                    self.assertIn('/' + detector + '/', model['path'])
                    arm = {'model_path': model['path'], 'model_sha256': model['sha256'],
                           'detector': detector, 'variant': variant, 'seed': 7, 'dataset':'SEA_A'}
                    self.assertEqual(worker.load_frozen(arm)['detector'], detector)
                    arm['detector'] = DETECTORS[(DETECTORS.index(detector) + 1) % 4]
                    with self.assertRaisesRegex(ValueError, 'detector'):
                        worker.load_frozen(arm)

    def test_phase_four_dispatches_all_24_models(self):
        config = dict(p.read(p.ROOT / 'configs/thesis.json'), dataset='SEA_A')
        def fake(root, config, seed, variant, detector):
            return {'detector': detector, 'variant': variant, 'seed': seed, 'dataset':'SEA_A'}
        with tempfile.TemporaryDirectory(dir=p.ROOT / 'tests') as tmp, \
             patch.object(training, 'ProcessPoolExecutor', ThreadPoolExecutor), \
             patch.object(training, 'train_one', side_effect=fake):
            models = training.train(Path(tmp), config, 2)
            actual = {(m['detector'], m['variant'], m['seed']) for m in models}
            expected = {(d, v, s) for d in DETECTORS for v in ['alarm', 'alarm_persistence'] for s in [7, 17, 27]}
            self.assertEqual(actual, expected)
            self.assertEqual(len(models), 24)

    def test_evaluation_arms_keep_the_trained_detector(self):
        models = [{'dataset':'SEA_A', 'detector': d, 'variant': v, 'seed': s, 'path': 'unused', 'sha256': 'unused'}
                  for d in DETECTORS for v in ['alarm', 'alarm_persistence'] for s in [7, 17, 27]]
        for detector in DETECTORS:
            arms = worker.comparison_arms({'table': worker.policy_table('0101')}, models, detector, 'SEA_A')
            self.assertEqual(len(arms), 10)
            self.assertEqual({a['detector'] for a in arms}, {detector})
        generic = p.arms(dict(p.read(p.ROOT / 'configs/thesis.json'), rl_models=models))
        self.assertEqual({a['detector'] for a in generic if p.is_rl(a['policy'])}, set(DETECTORS))

    def test_reference_pairs_do_not_cross_detectors(self):
        rows = []
        for detector, baseline in [('adwin', .8), ('hddm_w', .4)]:
            for arm, value in [('confirmed', baseline), ('error_rule', baseline + .1)]:
                row = dict(detector=detector, condition='same', stream='SEA_A', mode='clean', level='none',
                           arm=arm, feedback_fraction=.05, delay_blocks=1)
                row.update({metric: value for metric in analysis.METRICS}); rows.append(row)
        import pandas as pd
        with tempfile.TemporaryDirectory(dir=p.ROOT / 'tests') as tmp:
            analysis.report(rows, Path(tmp))
            pairs = pd.read_csv(Path(tmp) / 'paired_vs_confirmed.csv')
            for delta in pairs[pairs.arm.eq('error_rule')].host_macro_f1_difference:
                self.assertAlmostEqual(delta, .1)


if __name__ == '__main__':
    unittest.main()

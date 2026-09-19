"""Prepare the declared seed grid while preserving hashed execution provenance."""
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import protocol as p
from common.replication import effective_config, counts
runner=p.load_module('replication_runner','scripts/run_research.py')
OUT=runner.RESULTS


def main():
    spec=p.read(p.ROOT/'configs/study.json')
    marker=OUT/'replication_preparation.json'
    if marker.exists() and p.read(marker).get('status')=='complete':
        if p.read(OUT/'protocol.json')['spec']!=spec:raise ValueError('Prepared seed grid differs')
        print('Declared seed grid already prepared');return
    old_digest='320a54304c69331c931767b69d8f9eae77d4a395ebc0c61f160ffc912f0badcd'
    archive=p.ROOT/'provenance/execution_contracts'/old_digest
    records=archive/'records/results/per_dataset_study'
    old=p.read(records/'protocol.json')
    assert p.sha(records/'protocol.json')==old_digest
    changed={'01_attacks/prepare.py','01_attacks/per_dataset.py','04_rl_training/train.py',
             '05_experiment/policy_robustness.py','scripts/run_research.py','06_analysis/reporting/per_dataset_report.py'}
    for name,digest in old['sources'].items():
        assert p.sha(archive/'source'/name)==digest,name
        if name not in changed:assert p.sha(p.ROOT/name)==digest,name
    prior=old['spec']; proposed=p.read(p.ROOT/'configs/study.json')
    for key in ['rl_seeds','evaluation_seeds']:
        assert set(prior[key]).issubset(proposed[key]) and len(proposed[key])==5
    assert set(prior['radar']['evaluation_attack_assignments']).issubset(proposed['radar']['evaluation_attack_assignments'])
    import copy
    a,b=copy.deepcopy(prior),copy.deepcopy(proposed)
    for obj in [a,b]:
        obj.pop('rl_seeds');obj.pop('evaluation_seeds');obj['radar'].pop('evaluation_attack_assignments')
    assert a==b,'Only replication lists may change'
    if not marker.exists():
        assert p.sha(OUT/'protocol.json')==old_digest
        assert p.read(OUT/'completion.json')['status']=='complete'
        p.write(marker,{'status':'preparing','archive':archive.relative_to(p.ROOT).as_posix(),'started_utc':p.now()})
        for name in ['protocol.json','models.json','selected_policies.json','completion.json','report_manifest.json',
                     'training_status.json','validation_status.json','evaluation_status.json','sensitivity_status.json','launch.json']:
            path=OUT/name
            if path.exists():
                assert (records/name).exists(),name
                path.unlink()
        # Keep incomplete aggregate outputs from being mistaken for the final grid.
        for path in (OUT/'analysis').rglob('*'):
            if path.is_file():
                assert (records/path.relative_to(OUT)).exists(),str(path)
                path.unlink()
    radar=runner.freeze(spec)
    for role in ['training','validation','evaluation']:
        for dataset in spec['datasets']:
            source=OUT/'stages'/role/dataset
            saved=p.read(records/'stages'/role/dataset/'request.json')
            config=runner.configuration(spec,role,dataset)
            assert effective_config(saved['config'],role=='training')==effective_config(config,role=='training')
            manifest=runner.prepare.verify(source)
            if role=='evaluation':manifest=runner.prepare.build(source,config,radar if dataset=='radar' else None)
            p.write(source/'request.json',{'config':config,'protocol_sha256':p.sha(OUT/'protocol.json')})
            print(f'Prepared {role}/{dataset}: {len(manifest["conditions"])} conditions',flush=True)
    p.write(marker,{'status':'complete','archive':archive.relative_to(p.ROOT).as_posix(),
                    'counts':counts(spec),'completed_utc':p.now()})


if __name__=='__main__':main()

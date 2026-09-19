"""Compare the supplied RADAR feature bank with its published release.

Reads archived data and scripts as evidence; never executes upstream scripts or
changes research inputs. Supply the separately downloaded release ZIP with --archive.
"""
from pathlib import Path
import io
import argparse
import hashlib
import json
import sys
import zipfile
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from common import protocol as p
SOURCE=ROOT/'results/narrative/references/radar_release'
OUT=ROOT/'results/repository_checks/radar_provenance'


def raw_frequency(archive,clean,meta):
    with zipfile.ZipFile(archive) as outer:
        member=next(n for n in outer.namelist() if n.endswith('/Drift-dataset-raw.zip'))
        with zipfile.ZipFile(io.BytesIO(outer.read(member))) as data:
            digest=hashlib.sha256()
            with data.open('Drift-dataset.csv') as source:
                while chunk:=source.read(1024*1024):digest.update(chunk)
            with data.open('Drift-dataset.csv') as source:
                raw=pd.read_csv(source,usecols=['process.executable','process.parent.executable','target','phase'])
    proc=raw['process.executable'].str.split('\\').str[-1]
    parent=raw['process.parent.executable'].str.split('\\').str[-1]
    reconstructed=(proc.map(proc.value_counts())/(parent.map(parent.value_counts())+1)).to_numpy()
    bank=pq.read_table(clean,columns=['f62','label']).to_pandas()
    match=np.isclose(reconstructed,bank.f62,rtol=0,atol=5e-7)
    labels=raw.target.map(meta['label_mapping']).to_numpy()
    assert len(raw)==484753 and bool(match.all()) and np.array_equal(labels,bank.label)
    phase=raw.phase.to_numpy()
    p.write(OUT/'raw_frequency_check.json',{
        'status':'global_frequency_reconstruction_consistent_with_supplied_values',
        'checked_utc':p.now(),'raw_rows':len(raw),'raw_csv_sha256':digest.hexdigest(),
        'within_absolute_tolerance_5e_7':int(match.sum()),'all_rows_within_5e_7':bool(match.all()),
        'max_abs_difference':float(np.max(np.abs(reconstructed-bank.f62))),
        'reconstruction_interpretation':'Global full-stream process frequencies reconstruct every supplied ratio within 5e-7 absolute error. Exact decimal rounding of the supplied export was not assumed.',
        'phase_counts':{str(k):int(v) for k,v in raw.phase.value_counts().items()},
        'phase_boundaries':(np.flatnonzero(phase[1:]!=phase[:-1])+1).tolist(),
        'executed_estimated_splice_boundaries':meta['segment_boundaries'],
        'mapped_raw_labels_match':True,'raw_members_are_construction_phases':True,
        'used_for_original_event_metrics':False,'frozen_inputs_changed':False,
        'inspection_script_sha256':p.sha(Path(__file__))})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True, type=Path)
    args=parser.parse_args()
    archive=args.archive
    OUT.mkdir(parents=True,exist_ok=True)
    release=p.read(SOURCE/'manifest.json')
    assert p.sha(archive)==release['archive_sha256']
    catalogue=ROOT/'results/evaluation/00_streams/catalogue.json'
    item=p.read(catalogue)['radar'];meta=item['metadata']
    clean=(catalogue.parent/item['clean']).resolve()
    assert p.sha(clean)==meta['sha256']
    features=list(meta['feature_lineage'])
    columns={source:[v['source_column'] for v in meta['feature_lineage'].values() if v['source']==source]
             for source in ['fasttext','engineered']}
    rows=0;exact_cells=0;total_cells=0;maximum_error=0.0;labels_match=True
    with zipfile.ZipFile(archive) as outer:
        member=next(n for n in outer.namelist() if n.endswith('/dataset-drift.zip'))
        with zipfile.ZipFile(io.BytesIO(outer.read(member))) as data:
            digests={}
            for name in data.namelist():
                digest=hashlib.sha256()
                with data.open(name) as stream:
                    while chunk:=stream.read(1024*1024):digest.update(chunk)
                digests[name]=digest.hexdigest()
            with data.open('dataset-drift-fasttext.csv') as fast_file, data.open('dataset-drift-engineered.csv') as eng_file:
                fast=pd.read_csv(fast_file,chunksize=10000)
                eng=pd.read_csv(eng_file,chunksize=10000)
                bank=pq.ParquetFile(clean).iter_batches(batch_size=10000)
                for a,b,saved in zip(fast,eng,bank,strict=True):
                    original=saved.to_pandas()
                    assert len(a)==len(b)==len(original)
                    values=np.column_stack([a[columns['fasttext']].to_numpy(float),b[columns['engineered']].to_numpy(float)])
                    values=np.where(np.isfinite(values),values,0.0)
                    actual=original[features].to_numpy(float)
                    exact_cells+=int(np.equal(values,actual).sum());total_cells+=values.size
                    maximum_error=max(maximum_error,float(np.max(np.abs(values-actual))))
                    np.testing.assert_allclose(values,actual,atol=1e-12,rtol=1e-12)
                    target=a['target-class-name'].map(meta['label_mapping']).to_numpy()
                    labels_match=labels_match and bool(np.array_equal(target,original.label.to_numpy()))
                    rows+=len(original)
    assert rows==484753 and labels_match
    with zipfile.ZipFile(archive) as outer:
        member=next(n for n in outer.namelist() if n.endswith('/engineer-features.ipynb'))
        notebook=json.loads(outer.read(member))
    code='\n'.join(''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type']=='code')
    assert "df['process.name'].value_counts()" in code
    assert "df['process.parent.name'].value_counts()" in code
    assert meta['feature_lineage']['f62']['source_column']=='process_vs_parent_freq_ratio'
    excluded=[x['column'] for x in meta['excluded_metadata']]
    assert '@timestamp' in excluded
    report={'status':'published_feature_values_matched','checked_utc':p.now(),'rows':rows,'predictors':len(features),
        'published_release_doi':release['doi'],'archive_sha256':release['archive_sha256'],
        'published_csv_sha256':digests,'local_clean_sha256':p.sha(clean),
        'compared_numeric_cells':total_cells,'exact_numeric_cells':exact_cells,
        'maximum_absolute_numeric_difference':maximum_error,'comparison_atol':1e-12,'comparison_rtol':1e-12,
        'mapped_labels_match':labels_match,'upstream_code_executed':False,
        'timestamp_in_predictors':False,'source_is_assembled_lab_stream':True,
        'upstream_frequency_feature':{'local_feature':'f62','source_column':'process_vs_parent_freq_ratio',
            'notebook_computation':'Whole-input-dataframe process and parent value_counts, mapped to each row; ratio uses parent count + 1.',
            'online_reconstruction_performed':False,
            'implication':'Treat supplied feature vectors as offline inputs; causal raw-log feature availability is not established.'},
        'fasttext_fit_code_in_release':False,
        'inspection_script_sha256':p.sha(Path(__file__)),
        'remaining_limits':['FastText training corpus and timing are not established.',
            'The feature frequency statistic was supplied, not rebuilt from past-only observations.',
            'Timestamp amendments in the published script depend on class changes; timestamps are excluded from this experiment.',
            'One assembled lab stream does not establish independent deployment generalisation.']}
    p.write(OUT/'provenance_check.json',report)
    raw_frequency(archive,clean,meta)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()

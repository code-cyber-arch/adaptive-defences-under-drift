"""A standalone clone must not depend on files outside its repository root."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch
import hashlib
from common import protocol as p

prepare=p.load_module('reproduction_partition_test','01_attacks/per_dataset.py')
creator=p.load_module('reproduction_workspace_test','scripts/create_run.py')

class ReproductionPathTests(TestCase):
    def test_snapshot_uses_verified_local_input_without_parent_repository(self):
        with TemporaryDirectory() as directory:
            root=Path(directory); folder=root/'data/clean/RADAR';folder.mkdir(parents=True)
            data=folder/'full_stream.parquet';data.write_bytes(b'verified input fixture')
            p.write(folder/'metadata.json',{'rows':12,'sha256':p.sha(data),'path':'absent/original/location.parquet'})
            with patch.object(p,'ROOT',root):
                result=prepare.snapshot_radar({'radar':{'source_metadata':'data/clean/RADAR/metadata.json','source_rows':12}})
            self.assertEqual(result['snapshot'],'data/clean/RADAR/full_stream.parquet')
            self.assertEqual(result['source_sha256'],p.sha(data))

    def test_workspace_rejects_wrong_input_before_creating_output(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);folder=root/'data/clean/RADAR';folder.mkdir(parents=True)
            p.write(folder/'metadata.json',{'sha256':hashlib.sha256(b'correct').hexdigest()})
            wrong=root/'wrong.parquet';wrong.write_bytes(b'wrong')
            with patch.object(creator,'ROOT',root):
                with self.assertRaisesRegex(ValueError,'hash differs'):
                    creator.create('reproduction',wrong)
            self.assertFalse((root/'runs/reproduction').exists())

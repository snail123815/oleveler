import os
import re
import unittest

import pandas as pd

from oleveler.safe_annotations import safeExperimentNameMQ


class Test_safeExperimentName(unittest.TestCase):

    def setUp(self) -> None:
        self.MQpath = 'tests/MaxQuant_output_partial'
        self.MQpath_corr = 'tests/MaxQuant_output_partial_corr'
        self.infoFile = 'tests/annotation.csv'
        return super().setUp()

    def tearDown(self) -> None:
        filesMQoutput = []
        for p in [self.MQpath, self.MQpath_corr]:
            filesMQoutput.extend([os.path.join(p, f) for f in os.listdir(p)])
        for f in filesMQoutput:
            if f.endswith('._bk'):
                changedFile = f[:-4]
                assert changedFile in filesMQoutput
                os.remove(changedFile)
                os.rename(f, changedFile)
        return super().tearDown()

    def test_safeExperimentNameMQ(self):
        ret = safeExperimentNameMQ(self.MQpath)
        self.assertIsNone(ret)
        filesMQoutput = os.listdir(self.MQpath)
        self.assertIn('evidence.txt._bk', filesMQoutput)
        safeExperimentNameMQ(self.MQpath_corr)
        self.assertNotIn('evidence.txt._bk', os.listdir(self.MQpath_corr))
        # make sure: if changed, the file is still valid
        # no additional index column added
        evDf = pd.read_csv(os.path.join(self.MQpath, 'evidence.txt'), sep='\t', index_col=None)
        self.assertEqual(evDf.iloc[:,0].name, 'Sequence')
        illegal = re.compile('-+')
        experiments = evDf['Experiment'].unique()
        for exp in experiments:
            self.assertFalse(illegal.search(exp))
        
        # make sure the colum names match Experiments in evidence file.
        with open(os.path.join(self.MQpath, 'proteinGroups.txt'), 'r') as pg:
            columns = pg.readline().strip().split('\t')
        counts = 0
        for col in columns:
            for exp in experiments:
                if exp in col:
                    counts += 1
        self.assertEqual(counts, len(experiments)*8)


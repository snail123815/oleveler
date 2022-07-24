from oleveler import safeExperimentNameMQ

import unittest

import os
import shutil



class Test_safeExperimentName(unittest.TestCase):

    def test_safeExperimentNameMQ(self):
        MQpath = 'tests/MaxQuant_output_partial'
        ret = safeExperimentNameMQ(MQpath)
        self.assertIsNone(ret)
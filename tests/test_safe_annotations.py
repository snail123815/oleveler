import unittest
import pandas as pd
from pathlib import Path
from tempfile import NamedTemporaryFile
from oleveler.safe_annotations import safeAnnotations

class TestSafeAnnotations(unittest.TestCase):

    def setUp(self):
        # Create a temporary annotation file
        self.temp_file = NamedTemporaryFile(delete=False, suffix=".csv", mode="wt")
        self.temp_file.write("Raw.file,Condition,BioReplicate,Experiment\n")
        self.temp_file.write("file1,cond1,rep1,exp1\n")
        self.temp_file.write("file2,cond2,rep2,exp2\n")
        self.temp_file.write("file3,cond3,rep3,exp-3\n")
        self.temp_file.close()
        self.annotation_path = Path(self.temp_file.name)
        self.result_path = None

    def tearDown(self):
        # Remove the temporary file
        Path(self.temp_file.name).unlink()
        if self.result_path:
            Path(self.result_path).unlink()

    def test_safe_annotations_no_removal(self):
        result_path = safeAnnotations(self.annotation_path)
        result_df = pd.read_csv(result_path, index_col="Raw.file")
        self.assertIn("cond1", result_df["Condition"].values)
        self.assertIn("exp1", result_df["Experiment"].values)
        self.assertIn("exp_3", result_df["Experiment"].values)
        self.result_path = result_path

    def test_safe_annotations_with_removal(self):
        result_path = safeAnnotations(self.annotation_path, toRemove=["exp1"])
        result_df = pd.read_csv(result_path, index_col="Raw.file")
        self.assertNotIn("exp1", result_df["Experiment"].values)
        self.assertIn("exp2", result_df["Experiment"].values)
        self.assertIn("exp_3", result_df["Experiment"].values)
        self.result_path = result_path

    def test_safe_annotations_duplicate_index(self):
        with NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            temp_file.write(b"Raw.file,Condition,BioReplicate,Experiment\n")
            temp_file.write(b"file1,cond1,rep1,exp1\n")
            temp_file.write(b"file1,cond2,rep2,exp2\n")
            temp_file.close()
            annotation_path = Path(temp_file.name)

        with self.assertRaises(ValueError):
            safeAnnotations(annotation_path)

        Path(temp_file.name).unlink()

if __name__ == "__main__":
    unittest.main()
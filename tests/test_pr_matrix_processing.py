import unittest
import pandas as pd
from pathlib import Path
import shutil
from oleveler.limited_proteolysis.pr_matrix_processing import ProtPeps


class TestProtPeps(unittest.TestCase):
    def setUp(self):
        self.pr_matrix_path = Path(
            "tests/DIANN_output_partial/report.pr_matrix_r.tsv"
        )
        self.lcmsms_randomasiation_path = Path(
            "tests/DIANN_output_partial/lcmsms_randomisation.tsv"
        )
        self.output_path = Path("tests/DIANN_output_partial/dataTables")
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.pr_df = pd.read_csv(
            self.pr_matrix_path,
            sep="\t",
            header=0,
        )
        self.prot_peps = ProtPeps(
            name_experiment="test_experiment",
            pr_matrix_path=self.pr_matrix_path,
            experiment_mapper=self.lcmsms_randomasiation_path,
            sample_id_regex=r"-(DU-[\d]{3})_",
            output_path=self.output_path,
        )

    def tearDown(self):
        if self.output_path.exists():
            shutil.rmtree(self.output_path)
        return super().tearDown()

    def test_init(self):
        assert self.prot_peps.name == "test_experiment"
        assert self.prot_peps.pr_matrix_path == self.pr_matrix_path
        assert self.prot_peps.experiment_mapper == self.lcmsms_randomasiation_path
        assert self.prot_peps.sample_id_regex.pattern == r"-(DU-[\d]{3})_"
        assert self.prot_peps.ha == "9e069f", self.prot_peps.ha
        pep_path = self.output_path / "report_9e069f.pep_matrix_r.tsv"
        assert pep_path.exists(), list(self.output_path.iterdir())

    def test_reinit(self):
        self.prot_peps = ProtPeps(
            name_experiment="test_experiment",
            pr_matrix_path=self.pr_matrix_path,
            experiment_mapper=self.lcmsms_randomasiation_path,
            sample_id_regex=r"-(DU-[\d]{3})_",
            output_path=self.output_path,
        )

    def test_generate_mapper(self):
        result_mapper = self.prot_peps.__generate_mapper(self.pr_df)
        index_values = result_mapper.index.values
        assert len(index_values) == len(set(index_values))

    def test_generate_mapper_with_duplicates(self):
        pr_df_with_duplicates = pd.DataFrame(
            {
                "Protein.Group": ["P1", "P2", "P1", "P3", "P2", "P1"],
                "Stripped.Sequence": [
                    "SEQ1",
                    "SEQ2",
                    "SEQ1",
                    "SEQ3",
                    "SEQ4",
                    "SEQ1",
                ],
            }
        )
        expected_mapper = pd.Series(
            data=["P1", "P2", "P3", "P2"],
            index=["SEQ1", "SEQ2", "SEQ3", "SEQ4"],
            name="Protein.Group",
        )
        result_mapper = self.prot_peps._ProtPeps__generate_mapper(
            pr_df_with_duplicates
        )
        pd.testing.assert_series_equal(result_mapper, expected_mapper)


if __name__ == "__main__":
    unittest.main()

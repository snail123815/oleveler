import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from oleveler.main import calHash

# columns in volcano plot data
MLOG10ADJP_COL = "adj.pvalue"
LOG2FC_COL = "log2FC"


class ProtPeps:
    def __init__(
        self,
        name_experiment,
        pr_matrix_path,
        experiment_mapper,
        sample_id_regex,
    ) -> None:

        self.name = name_experiment
        if isinstance(pr_matrix_path, str):
            self.pr_matrix_path = Path(pr_matrix_path)
        else:
            self.pr_matrix_path = pr_matrix_path
        if isinstance(experiment_mapper, str):
            self.experiment_mapper = Path(experiment_mapper)
        else:
            self.experiment_mapper = experiment_mapper
        if isinstance(sample_id_regex, str):
            self.sample_id_regex = re.compile(sample_id_regex)
        else:
            self.sample_id_regex = sample_id_regex

        self.ha = calHash(
            self.pr_matrix_path, self.experiment_mapper, self.sample_id_regex
        )

        self.pr_df, self.pep_df, self.pep_matrix_output = (
            self.__process_pr_to_pep()
        )
        self.protPep_mapper = self.__generate_mapper(self.pr_df)

    __CAN_CONVERT_TABLE_FILE_EXTENSIONS = [".tsv", ".xlsx"]

    def __generate_mapper(self, pr_df) -> pd.Series:
        """
        Generates a mapper from protein groups to stripped sequences.

        This function concatenates the 'Protein.Group' and 'Stripped.Sequence' columns
        of the provided DataFrame twice, drops duplicates, and sets the 'Stripped.Sequence'
        as the index. It returns a Series mapping 'Stripped.Sequence' to 'Protein.Group'.

        Args:
            pr_df (pd.DataFrame): DataFrame containing protein group and peptide information.

        Returns:
            pd.Series: A Series mapping 'Stripped.Sequence' to 'Protein.Group'.
        """
        protein_peptide_mapper = (
            pd.concat(
                [
                    pd.concat(
                        [
                            pr_df["Protein.Group"],
                            pr_df["Stripped.Sequence"],
                        ],
                        axis=1,
                    ),
                    pd.concat(
                        [
                            pr_df["Protein.Group"],
                            pr_df["Stripped.Sequence"],
                        ],
                        axis=1,
                    ),
                ],
                axis=0,
                ignore_index=True,
            )
            .drop_duplicates()
            .set_index("Stripped.Sequence")["Protein.Group"]
        )

        # Remove some remaining razer sequence
        # Stripped.Sequence
        # CIGCHTCSVTCK           SCO0217;SCO6534
        # CIGCHTCSVTCK                   SCO0217
        # KLMSWVDEEA             SCO5514;SCO7154
        # KLMSWVDEEA                     SCO5514
        # KVAVIGYGSQGHAHALSLR            SCO5514
        # KVAVIGYGSQGHAHALSLR    SCO5514;SCO7154

        duplicated_sequences = {}
        for seq, pg in protein_peptide_mapper[
            protein_peptide_mapper.index.duplicated(keep=False)
        ].items():
            if seq not in duplicated_sequences:
                duplicated_sequences[seq] = pg
            else:
                if len(pg) > len(duplicated_sequences[seq]):
                    duplicated_sequences[seq] = pg
        protein_peptide_mapper = pd.concat(
            [
                protein_peptide_mapper[
                    protein_peptide_mapper.index.drop_duplicates(keep=False)
                ],
                pd.Series(duplicated_sequences),
            ]
        )
        assert protein_peptide_mapper[
            protein_peptide_mapper.index.duplicated(keep=False)
        ].index.shape == (0,)

        return protein_peptide_mapper

    def __read_pr_matrix(self) -> pd.DataFrame:
        """
        Sample names from analysis output in pr_matrix_path is Burker data .d format
        """
        pr_df = pd.read_csv(self.pr_matrix_path, sep="\t", header=0)
        metadata_columns = []
        data_columns = []
        for c in pr_df.columns:
            if c.endswith(".d"):
                data_columns.append(c)
            else:
                metadata_columns.append(c)

        data_run_ids = []
        for c in data_columns:
            try:
                data_run_ids.append(self.sample_id_regex.search(c).group(1))
            except AttributeError as e:
                print(data_run_ids)
                print(c)
                print(self.sample_id_regex)
                raise e

        randomisation_data = pd.read_csv(
            self.experiment_mapper, sep="\t", index_col=1, header=None
        )[0]
        try:
            named_data_columns = [
                randomisation_data[id].replace(".", "_") for id in data_run_ids
            ]
        except:
            raise ValueError(f"There is error in regex matching ids.")
        sorted_data_columns = sorted(named_data_columns)

        pr_df = pd.concat(
            [
                pr_df.loc[:, metadata_columns],
                pd.DataFrame(
                    pr_df.loc[:, data_columns].values,
                    index=pr_df.index,
                    columns=named_data_columns,
                ),
            ],
            axis=1,
        )
        pr_df = pr_df.loc[:, metadata_columns + sorted_data_columns]
        return pr_df, sorted_data_columns

    def __sum_pr_to_pep(self, pr_df, data_columns) -> pd.DataFrame:
        """
        Some peptides are detected multiple times with different charges
        or with variable modifications. For now, when we care only about
        the quantity of the peptide, all different status needs to be summed.
        """
        pep_data = {}
        current_pep = ""
        current_data = np.zeros((len(data_columns),))
        pr_df_copy = pr_df.sort_values("Stripped.Sequence", ascending=True)

        for _, row in tqdm(pr_df_copy.iterrows(), total=pr_df_copy.shape[0]):
            data = np.array(row[data_columns], dtype=float)
            if row["Stripped.Sequence"] != current_pep:
                # Save previous result
                pep_data[current_pep] = current_data
                # Reset and give values
                current_pep = row["Stripped.Sequence"]
                current_data = data
            else:
                current_data += data
        # save the last peptide
        pep_data[current_pep] = current_data
        pep_data.pop("")  # remove the first empty pep
        # convert to dataframe
        return pd.DataFrame.from_dict(
            pep_data, orient="index", columns=data_columns
        ).round(3)

    def __process_pr_to_pep(self):
        """
        DIANN output pr_matrix
        pr_matrix_path = Path("./diann_result_lip_202405221001_<hash>.pr_matrix.tsv")
        # Data produced by DIA-NN, quantification data in columns with name of
        # raw data file, which ends with ".d", which is Bruker files
        experiment_mapper = Path("./lcmsms_randomisation.tsv")
        # Two columns with no index nor header
        # First column = experiment names, no duplicates (with number indicating reps)
        # Second column = match of the following sample id from the raw data file names.
        sample_id_regex = re.compile(r"-(DU-[\\d]{3})_")
        """
        assert ".pr_matrix" in str(
            self.pr_matrix_path
        ), f"String '.pr_matrix' not found in {self.pr_matrix_path}"

        pep_matrix_output = Path(
            "./dataTables/"
            + self.pr_matrix_path.name.replace(
                ".pr_matrix", f"_{self.ha}.pep_matrix"
            )
        )
        pr_df, data_columns = self.__read_pr_matrix()

        if pep_matrix_output.exists():
            pep_df = pd.read_csv(
                pep_matrix_output, sep="\t", header=0, index_col=0
            )
        else:
            pep_df = self.__sum_pr_to_pep(pr_df, data_columns)
            pep_df.to_csv(pep_matrix_output, sep="\t")

        print(f"{self.name} peptide matrix shape: {pep_df.shape}")
        print(f"{self.name} peptide matrix path: {pep_matrix_output}")

        return pr_df, pep_df, pep_matrix_output

    def __map_proteingroup_to_peptide_df(
        self,
        pep_indexed_df: pd.DataFrame,
    ):
        (mapped,) = (pep_indexed_df.index.to_series().map(self.protPep_mapper),)
        if all(pd.isna(m) for m in mapped):
            # print(f'Mapping not successful, return original table.')
            return None
        target_df = pd.concat(
            [
                mapped,
                pep_indexed_df,
            ],
            axis=1,
        )
        target_df.index.name = "Peptide"
        target_df.columns = ["Protein_group"] + target_df.columns[1:].tolist()
        return target_df

    def map_proteingroup_to_peptide(self, table_path) -> pd.DataFrame:
        if isinstance(table_path, str):
            table_path = Path(table_path)

        assert table_path.suffix in self.__CAN_CONVERT_TABLE_FILE_EXTENSIONS, (
            f"{table_path} does not have recognisable extension, "
            f"must be one of {self.__CAN_CONVERT_TABLE_FILE_EXTENSIONS}"
        )

        if table_path.suffix == ".tsv":
            pep_indexed_df = pd.read_csv(table_path, sep="\t", index_col=0)
        elif table_path.suffix == ".xlsx":
            try:
                pep_indexed_df = pd.read_excel(table_path, index_col=0)
            except ValueError as excel_read_err:
                print(f"Excel file {table_path} read failed")
                raise excel_read_err
        target_df = self.__map_proteingroup_to_peptide_df(pep_indexed_df)
        if target_df is not None:
            print(f"Mapped {table_path}")
        else:
            print(f"Not mapped: {table_path}")
        return target_df

    def mapdir_proteingroup_to_peptide(
        self, dir_path: Path, overwrite=False
    ) -> None:
        mapped_dir = Path(f"{dir_path}_mapped")
        for target_file in dir_path.iterdir():
            if (
                target_file.suffix
                not in self.__CAN_CONVERT_TABLE_FILE_EXTENSIONS
            ):
                continue
            if target_file.stem.endswith("_mapped"):
                continue
            mapped_file = mapped_dir / f"{target_file.stem}_mapped.tsv"
            if not overwrite:
                if mapped_file.exists():
                    continue
            target_df = self.map_proteingroup_to_peptide(target_file)
            if target_df is not None:
                mapped_dir.mkdir(exist_ok=True)
                print(f"\toutput: {mapped_file}")
                target_df.to_csv(mapped_file, sep="\t")

    def get_all_peptides(self, protein_group: str) -> list[str]:
        """
        Returns a list of all peptides for the protein in the peptide matrix.
        """
        assert (
            protein_group in self.protPep_mapper.values
        ), f"Protein group {protein_group} not found in the peptide matrix."
        return sorted(
            self.protPep_mapper[
                self.protPep_mapper == protein_group
            ].index.tolist()
        )


def filter_volcano(
    volcano_df,
    log2fc_t=1,
    minus_logp_t=-np.log10(0.05),
    overlap_check_range=100,
    target_protein="SCO4648",
):

    minus_logp_t = float(minus_logp_t)
    for c in volcano_df.columns:
        if c.startswith(MLOG10ADJP_COL):
            MLOG10ADJP_COL = c
        elif c.startswith(LOG2FC_COL):
            LOG2FC_COL = c
        else:
            pass
    filtered = volcano_df[
        (volcano_df[MLOG10ADJP_COL] >= minus_logp_t)
        & (
            (volcano_df[LOG2FC_COL] >= log2fc_t)
            | (volcano_df[LOG2FC_COL] <= -log2fc_t)
        )
    ].sort_values(LOG2FC_COL)
    filtered_up = filtered[(filtered[LOG2FC_COL] >= log2fc_t)].iloc[
        -overlap_check_range:, :
    ]
    filtered_down = filtered[(filtered[LOG2FC_COL] <= -log2fc_t)].iloc[
        :overlap_check_range, :
    ]
    overlap_pgs = []
    for pep, pg in filtered_up["Protein_group"].items():
        if pg in filtered_down["Protein_group"]:
            overlap_pgs.append((pep, pg))
    if len(overlap_pgs) > 0:
        print(
            "Found protein group(s) contain both up and down regulated peptides"
        )
        print(overlap_pgs)
    else:
        print("No protein group overlap for up and down regulated peptides")

    target_protein_in_filtered = filtered[
        filtered["Protein_group"].str.contains(target_protein)
    ]
    target_protein_in_all = volcano_df[
        volcano_df["Protein_group"].str.contains(target_protein)
    ]
    if target_protein_in_filtered.shape[0] > 0:
        print(
            "There are significant changes with the following thresholds:\n"
            f"\tFC >= {np.power(2,log2fc_t):.2f} (log2fc {log2fc_t:.2f})\n"
            f"\tp <= {np.power(10, -minus_logp_t):.2f}, (-log10(p) {minus_logp_t:.2f})"
        )
        print()
        print(target_protein_in_filtered)
        print("None significant changes:")
        print()
        print(
            pd.concat(
                [target_protein_in_all, target_protein_in_filtered],
                axis=0,
                # ignore_index=True,
            ).drop_duplicates(keep=False)
        )
    else:
        print(
            "There are NO significant changes with the following thresholds:\n"
            f"\tFC >= {np.power(2,log2fc_t):.2f} (log2fc {log2fc_t:.2f})\n"
            f"\tp <= {np.power(10, -minus_logp_t):.2f}, (-log10(p) {minus_logp_t:.2f})\n"
            "Print all:"
        )
        print()
        print(target_protein_in_all)


def filter_volcano(
    volcano_df,
    log2fc_t=1,
    minus_logp_t=-np.log10(0.05),
    overlap_check_range=None,  # Number of peptides to check around 0 for possible overlaps
    target_protein=None,
) -> tuple[pd.DataFrame, list, pd.DataFrame]:
    """
    return filtered, overlap_pgs, target_protein_filtered
    """
    global MLOG10ADJP_COL
    global LOG2FC_COL
    minus_logp_t = float(minus_logp_t)
    for c in volcano_df.columns:
        if c.startswith(MLOG10ADJP_COL):
            MLOG10ADJP_COL = c
        elif c.startswith(LOG2FC_COL):
            LOG2FC_COL = c
        else:
            pass

    filtered_up = volcano_df[
        (volcano_df[MLOG10ADJP_COL] >= minus_logp_t)
        & (volcano_df[LOG2FC_COL] >= log2fc_t)
    ].sort_values(LOG2FC_COL)
    filtered_down = volcano_df[
        (volcano_df[MLOG10ADJP_COL] >= minus_logp_t)
        & (volcano_df[LOG2FC_COL] <= -log2fc_t)
    ].sort_values(LOG2FC_COL)
    filtered = pd.concat([filtered_up, filtered_down], axis=0)

    overlap_pgs = []
    if overlap_check_range is None:
        ocr = max(filtered_up.shape[0], filtered_down.shape[0])
    else:
        ocr = min(
            max(filtered_up.shape[0], filtered_down.shape[0]),
            overlap_check_range,
        )
    for pepup, pg in filtered_up.iloc[-ocr:, :]["Protein_group"].items():
        if pg in filtered_down.iloc[:ocr, :]["Protein_group"]:
            peps_down = filtered_down[
                filtered_down.iloc[:ocr, :]["Protein_group"] == pg
            ]
            overlap_pgs.append((pg, pepup, peps_down))
    if len(overlap_pgs) > 0:
        print(
            "Found protein group(s) contain both up and down regulated peptides"
        )
        print(overlap_pgs)
    else:
        print(
            "No protein group overlap for up and down regulated peptides",
            (
                (
                    f"when checking +- {ocr} peptides (requested "
                    f"{overlap_check_range}) around zero"
                )
                if overlap_check_range is not None
                else ""
            ),
        )

    if target_protein is None:
        return filtered, overlap_pgs, None
    else:
        target_protein_filtered = filtered[
            filtered["Protein_group"].str.contains(target_protein)
        ]
        target_protein_in_all = volcano_df[
            volcano_df["Protein_group"].str.contains(target_protein)
        ]
        if target_protein_filtered.shape[0] > 0:
            print(
                "There are significant changes with the following thresholds:\n"
                f"\tFC >= {np.power(2,log2fc_t):.2f} (log2fc {log2fc_t:.2f})\n"
                f"\tp <= {np.power(10, -minus_logp_t):.2f}, (-log10(p) {minus_logp_t:.2f})"
            )
            print()
            print(target_protein_filtered)
            print("None significant changes:")
            print()
            print(
                pd.concat(
                    [target_protein_in_all, target_protein_filtered],
                    axis=0,
                    # ignore_index=True,
                ).drop_duplicates(keep=False)
            )
        else:
            print(
                "There are NO significant changes with the following thresholds:\n"
                f"\tFC >= {np.power(2,log2fc_t):.2f} (log2fc {log2fc_t:.2f})\n"
                f"\tp <= {np.power(10, -minus_logp_t):.2f}, (-log10(p) {minus_logp_t:.2f})\n"
                "Print all:"
            )
            print()
            print(target_protein_in_all)
        return filtered, overlap_pgs, target_protein_filtered

# 1. Find proteins that have distinguishable changed peptides in LiP samples
# 2. Exclude those have changed peptides in Trp samples
# 3. Adjust protein amount based on Trp sample, check literature MSstatsLiP
#    https://doi.org/10.1038/s41596-022-00771-x
# 4. Still changed? Bingo!


from pathlib import Path

import numpy as np
import pandas as pd

from oleveler.limited_proteolysis import ProtPeps, filter_pep_volcano_get_proteins


class Real_Diff_Peps_Finder:

    def __init__(
        self,
        exp_volcano_path: Path,  # for example LiP
        Exp_ProtPeps: ProtPeps,
        ctr_volcano_path: Path,  # for example Trp
        Ctr_ProtPeps: ProtPeps,
        # trp_protein_volcano_data_path: Path,
    ) -> None:
        """
        Get input volcano data,
        do (in the future) protein amount based adjustment
        """
        # map volcano plot
        self.exp_volcano_mapped_df = Exp_ProtPeps.map_proteingroup_to_peptide(
            exp_volcano_path
        )
        self.ctr_volcano_mapped_df = Ctr_ProtPeps.map_proteingroup_to_peptide(
            ctr_volcano_path
        )

    def get_sig_proteins(
        self,
        log2fc_t=np.log2(2),
        minus_logp_t=-np.log10(0.01),
        overlap_check_range=None,
    ):
        filtered_exp, _, _ = filter_pep_volcano_get_proteins(
            self.exp_volcano_mapped_df,
            log2fc_t,
            minus_logp_t,
            overlap_check_range,
        )
        filtered_ctr, _, _ = filter_pep_volcano_get_proteins(
            self.ctr_volcano_mapped_df,
            log2fc_t,
            minus_logp_t,
            overlap_check_range,
        )

        filtered_exp.rename(columns=lambda c: str(c) + "_exp", inplace=True)
        filtered_ctr.rename(columns=lambda c: str(c) + "_ctr", inplace=True)
        intersection_peps = filtered_exp.index.intersection(filtered_ctr.index)
        if len(intersection_peps) > 0:
            print("Intersection found")
            filtered_repeated = pd.merge(
                filtered_exp,
                filtered_ctr,
                "inner",
                left_index=True,
                right_index=True,
            )

            peps_to_remove = []
            for pep, data in filtered_repeated.iterrows():
                r_log2fc = abs(
                    (data["log2FC_exp"] - data["log2FC_ctr"])
                    / data["log2FC_ctr"]
                )
                r_transp = (
                    abs(data["adj.pvalue_exp"] - data["adj.pvalue_ctr"])
                    / data["adj.pvalue_ctr"]
                )
                if r_log2fc < log2fc_t and r_transp < minus_logp_t:
                    peps_to_remove.append(pep)

            filtered_exp = filtered_exp.loc[
                filtered_exp.index.difference(set(peps_to_remove)), :
            ]
        else:
            print("No intersection between exp and ctr.")

        # remove the appended _exp
        sig_peptides = filtered_exp.rename(columns=lambda c: c[:-4])

        sig_proteins = {}
        for p in sig_peptides["Protein_group"].unique():
            ups = sig_peptides.index[
                (sig_peptides["Protein_group"] == p)
                & (sig_peptides["log2FC"] > 0)
            ]
            ups_log2fc = sig_peptides.loc[ups, "log2FC"]
            ups_mlog10p = sig_peptides.loc[ups, "adj.pvalue"]
            downs = sig_peptides.index[
                (sig_peptides["Protein_group"] == p)
                & (sig_peptides["log2FC"] < 0)
            ]
            downs_log2fc = sig_peptides.loc[downs, "log2FC"]
            downs_mlog10p = sig_peptides.loc[downs, "adj.pvalue"]
            sig_proteins[p] = [
                len(ups),
                ";".join(ups),
                ";".join(ups_log2fc.astype(str)),
                ";".join(ups_mlog10p.astype(str)),
                len(downs),
                ";".join(downs),
                ";".join(downs_log2fc.astype(str)),
                ";".join(downs_mlog10p.astype(str)),
            ]
        sig_proteins = pd.DataFrame(
            sig_proteins,
            index=[
                "ups_n",
                "ups",
                "ups_log2fc",
                "ups_mlog10p",
                "downs_n",
                "downs",
                "downs_log2fc",
                "downs_mlog10p",
            ],
        ).T
        sig_proteins.sort_index(inplace=True)
        sig_proteins["total_n_sigs"] = (
            sig_proteins["ups_n"] + sig_proteins["downs_n"]
        )
        sig_proteins.sort_values(
            by=["total_n_sigs", "ups_n", "downs_n"],
            ascending=False,
            inplace=True,
        )
        sig_proteins.drop(columns="total_n_sigs", inplace=True)
        return sig_proteins, sig_peptides


def main():
    # ProtPeps_202309_lip = ProtPeps(
    #     "LiP_202309",
    #     "./data202309/fragpipe_diann_LiP/report.pr_matrix.tsv",
    #     "./lcmsms_randomisation_202309.tsv",
    #     r"(D[-]?[\d]{3})_",
    # )
    # ProtPeps_202309_trp =ProtPeps(
    #     "Trp_202309",
    #     "./data202309/fragpipe_diann_Trp/report.pr_matrix.tsv",
    #     "./lcmsms_randomisation_202309.tsv",
    #     r"(D[-]?[\d]{3})_",
    # )

    # ProtPeps_202403_lip = ProtPeps(
    #     "LiP_202403",
    #     "./data202403/fragpipe_diann_LiP/report.pr_matrix.tsv",
    #     "./lcmsms_randomisation_202403.tsv",
    #     r"-(DU-[\d]{3})_",
    # )
    # ProtPeps_202403_trp = ProtPeps(
    #     "Trp_202403",
    #     "./data202403/fragpipe_diann_Trp/report.pr_matrix.tsv",
    #     "./lcmsms_randomisation_202403.tsv",
    #     r"-(DU-[\d]{3})_",
    # )

    ProtPeps_202408_lip = ProtPeps(
        "LiP_202408",
        "./data202408/lip_samples/report.pr_matrix.tsv",
        "./lcmsms_randomisation_202408.tsv",
        r"-(DU-[\d]{3})_",
    )
    ProtPeps_202408_trp = ProtPeps(
        "Trp_202408",
        "./data202408/trp_samples_exclude017/report.pr_matrix.tsv",
        "./lcmsms_randomisation_202408.tsv",
        r"-(DU-[\d]{3})_",
    )

    rdf = Real_Diff_Peps_Finder(
        "Plots/Volcano/Volcano_DESeq2_shrink_comp_result_SA_Thio02_LiP_af6277.xlsx",
        ProtPeps_202408_lip,
        "Plots/Volcano/Volcano_DESeq2_shrink_comp_result_trp_SA_Thio02_Trp_6ed6bd.xlsx",
        ProtPeps_202408_trp,
    )

    sig_proteins, sig_peptides = rdf.get_sig_proteins(
        log2fc_t=np.log2(1.5),
        minus_logp_t=-np.log10(0.001),
        overlap_check_range=500,
    )

    print(sig_proteins)
    print(sig_peptides)


if __name__ == "__main__":
    main()

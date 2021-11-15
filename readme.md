# Oleveler

Title for Omics-leveler. Seems to deal with all omics data? No, only quantitive proteomics and transcriptomics.

The functions are desigend to be (mostly) directly callable from within jupyter notebook when imported as:

```python
from oleveler import *
```

(Thus one file)

Tested in jupyter lab only.

## 1. Install Dependencies

Create a conda environment for **oleveler** is recommanded. [**Mamba**](https://github.com/mamba-org/mamba) is recommanded to install dependencies because it is much faster and reliable than **conda** itself. To install **Mamba** in your conda environment:

```sh
conda install mamba -n base -c conda-forge
```

Then you can clone this repository to your local dir by:

```sh
git clone --depth 1 https://github.com/snail123815/oleveler.git
```

And start creating a environment with dependencies:

```sh
cd oleveler
mamba env create -n oleveler -f oleveler_deps.yml
```

No error message should appear.

Before running your analysis, do not forget to activate the environment you just created:

```sh
mamba activate oleveler
```

## 2. Prepare your data - Example folder `my_analysis`

The analysis needs to start with a fresh (empty) folder. Assume the folder is named `my_analysis`. Please **copy** the `oleveler.py` file from this repository (or download it using the download )

### 2.1 Proteomics data

Oleveler starts with [MaxQuant](https://maxquant.org) processed data. From the analysis folder `combined/txt/`, please copy the following two files:

- `evidence.txt`
- `proteinGroups.txt`

Create a folder named **`MaxQuant_output`** in `my_analysis` folder, put the above files in that folder.

#### 2.1.1 Edit `Annotation.csv`

Copy `Annotation_proteomics_example.csv` from this project, put it directly in `my_analysis` folder and rename it as `Annotation.csv`.

Open `Annotation.csv` with Excel, edit it to fit your proteomics project.

There are four columns in this file: `Raw.file`, `Condition`, `BioReplicate`, `Experiment`

`Raw.file` - fill in all the file name of the raw proteomics files, but without the file extension. Eg. for '210619_DC_01.raw' you can fill in '210619_DC_01'. Contents of this column needs to be unique.  
`Condition` - fill in the experimental condition for each raw file. Same condition (biological replicates should share the same name). Please add all experimental information to this column. Eg. strain 'WT' in 'MM' medium collected at 24 hours, you should enter something like 'WT_MM_24'.  
`BioReplicate` - fill in a number of the bio-replicate within one condition. Eg. '210619_DC_01', '210619_DC_09', and '210619_DC_20' are samples from the same condition, then give them numbering '1', '2', and '3' in this column. Orders do not matter.  
`Experiment` - fill in the experiment name of the raw file belongs. Each should contain both condition and bio-replicate information. **Contents of this column needs to be unique only if there is only one LC-MS/MS run per sample.**

This file will be used in **MSstats**<sup>[2][2]</sup> so it should comply with its [rules](https://msstats.org/wp-content/uploads/2020/02/MSstats_v3.18.1_manual_2020Feb26-v2.pdf). Although it is possible to use data with multiple runs per sample (eg. samples that fractionised before LC-MS/MS runs), but that have not been tested in **Oleveler**.

#### 2.1.2 Edit `comparisons.xlsx`

This file is to provide enough information for the program to do "different analysis" to see which proteins are changed in different conditions, the condition of experiment and corresponding control needs to be specified.

Copy `comparison_example.xlsx` to `my_analysis` folder and rename it as `comparison.xlsx`. Open it with Excel, edit it to fit your project.

There are three columns in this file: `id`, `exp`, `ctr`

`id` - identifier for this perticular comparison. Using this id you can let the notebook show specific analysis result (volcano plot etc.).  
`exp` - the **condition** that will be defined as 'experiment condition'. Needs to be one of the conditions listed in `Condition` column of `Annotation.csv` file.  
`ctr` - the **condition** that will be defined as 'control condition'.

The different analysis will show the 'log<sub>2</sub> fold changes' (LFCs) of `exp` divided by `ctr`. When zero is encountered in this comparision, result will show as `inf` for 'divided by zero' conditions, `-inf` for zero devide positive number conditons, otherwise `nan` for 'not a number'.

### 2.2 Transcriptomics data

Oleveler starts with feature counts (usually it is read counts for each gene) files. These files usually are genereated by [**featureCounts**](http://subread.sourceforge.net). ([**Salmon**](https://combine-lab.github.io/salmon/) support is on the way)

Create a folder named **`quantResult`** in `my_analysis` folder, put read counts files for all samples in this folder.

#### 2.2.1 Edit `Annotation.csv`

Copy `Annotation_example.csv` from this project, put it directly in `my_analysis` folder and rename it as `Annotation.csv`.

Open `Annotation.csv` with Excel, edit it to fit your proteomics project.

There are four columns in this file: `Raw.file`, `Condition`, `BioReplicate`, `Experiment`  
(if you do not see four columns, that means excel did not reconise the file as `csv` format, read instructions from [Microsoft](https://support.microsoft.com/en-us/office/import-or-export-text-txt-or-csv-files-5250ac4c-663c-47ce-937b-339e391393ba).)

`Raw.file` - fill in all the file name of the read counts files, but without the file extension. Eg. for 'D24_1.txt' you can fill in 'D24_1'. Contents of this column needs to be unique.  
`Condition` - fill in the experimental condition for each read counts file. Same condition (biological replicates should share the same name). Please add all experimental information to this column. Eg. strain 'WT' in 'MM' medium collected at 24 hours, you should enter something like 'WT_MM_24'.  
`BioReplicate` - fill in a number of the bio-replicate within one condition. Eg. 'D24_1', 'D24_2', and 'D24_3' are samples from the same condition, then give them numbering '1', '2', and '3' in this column. Orders do not matter.  
`Experiment` - fill in the experiment name of the raw file belongs. Each should contain both condition and bio-replicate information. **Contents of this column needs to be unique.**

#### 2.2.2 Edit `comparisons.xlsx`

This file is to provide enough information for the program to do "different analysis" to see which genes are changed in different conditions, the condition of experiment and corresponding control needs to be specified.

Copy `comparison_example.xlsx` to `my_analysis` folder and rename it as `comparison.xlsx`. Open it with Excel, edit it to fit your project.

There are three columns in this file: `id`, `exp`, `ctr`

`id` - identifier for this perticular comparison. Using this id you can let the notebook show specific analysis result (volcano plot etc.).  
`exp` - the **condition** that will be defined as 'experiment condition'. Needs to be one of the conditions listed in `Condition` column of `Annotation.csv` file.  
`ctr` - the **condition** that will be defined as 'control condition'.

The different analysis will show the 'log<sub>2</sub> fold changes' (LFCs) of `exp` divided by `ctr`. When zero is encountered in this comparision, result will show as `inf` for 'divided by zero' conditions, `-inf` for zero devide positive number conditons, otherwise `nan` for 'not a number'.

## 3. TODO:

- [x] Close temp files opened due to required compatibility to windows
- [x] \*DESeq2 will die when too many comparisons will run in the same R kernel. This often happens because each comparison consume to much memory and thus leads to memory surge or other problem. Solved by removing tempfiles etc.
- [x] Add queryLfc()
- [ ] Add queryViolin()
  - For query gene list
- [ ] Pack related functions
- [x] Add references list

## 4. References

1. Love, M.I., Huber, W. & Anders, S. Moderated estimation of fold change and dispersion for RNA-seq data with DESeq2. Genome Biol 15, 550 (2014)

[1]:https://doi.org/10.1186/s13059-014-0550-8 "DESeq2" 

2. Meena Choi, Ching-Yun Chang, Timothy Clough, Daniel Broudy, Trevor Killeen, Brendan MacLean, Olga Vitek, MSstats: an R package for statistical analysis of quantitative mass spectrometry-based proteomic experiments, Bioinformatics, Volume 30, Issue 17, 1 September 2014, Pages 2524–2526

[2]:https://doi.org/10.1093/bioinformatics/btu305 "MSstats"

3. Anqi Zhu, Joseph G Ibrahim, Michael I Love, Heavy-tailed prior distributions for sequence count data: removing the noise and preserving large differences, Bioinformatics, Volume 35, Issue 12, June 2019, Pages 2084–2092

[3]: https://doi.org/10.1093/bioinformatics/bty895 "apeglm"

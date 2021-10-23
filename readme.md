# Oleveler

Title for Omics-leveler. Seems to deal with all omics data? No, only quantitive proteomics and transcriptomics.

The functions are desigend to be (mostly) directly callable from within jupyter notebook when imported as:

```python
from oleveler import *
```

(Thus one file)

Tested in jupyter lab only.

Please include `%matplotlib widget` to allow interactive plotting. However, with this the plots cannot be saved together with `.ipynb` file. The updated jupyterlab module `...` will deal with that problem. For now, if you want to save plots with `.ipynb`, please use `%matplotlib inline`.

## Dependencies

(INCOMPLETE)

- mamba/conda/micromamba -c conda-forge -c bioconda -c r
  - numpy pandas scipy rpy2 matplotlib scikit-learn r openpyxl jupyterlab nodejs ipympl
- pip install
  - bcbio-gff
- R
  - MSstats
  - DESeq2
  - apeglm # for lfc shrinkage

## TODO:

- [x] Close temp files opened due to required compatibility to windows
- [x] \*DESeq2 will die when too many comparisons will run in the same R kernel. This often happens because each comparison consume to much memory and thus leads to memory surge or other problem. Solved by removing tempfiles etc.
- [ ] Pack related functions
- [ ] Add references list
  - apeglm<sup>[1][1]</sup>
  - DESeq2<sup>[2][2]</sup>
  - [MSstats](10.1093/bioinformatics/btu305)
  - []

1. Anqi Zhu, Joseph G Ibrahim, Michael I Love, Heavy-tailed prior distributions for sequence count data: removing the noise and preserving large differences, Bioinformatics, Volume 35, Issue 12, June 2019, Pages 2084–2092

[1]: https://doi.org/10.1093/bioinformatics/btyy895 "apeglm"

2. Love, M.I., Huber, W. & Anders, S. Moderated estimation of fold change and dispersion for RNA-seq data with DESeq2. Genome Biol 15, 550 (2014)

[2]:https://doi.org/10.1186/s13059-014-0550-8 "DESeq2" 

# Oleveler

Title for Omics-leveler. Seems to deal with all omics data? No, only quantitive proteomics and transcriptomics.

The functions are desigend to be (mostly) directly callable from within jupyter notebook when imported as:

```python
from oleveler import *
```

Tested in jupyter lab only.

Please include `%matplotlib widget` to allow interactive plotting. However, with this the plots cannot be saved together with `.ipynb` file. The updated jupyterlab module `...` will deal with that problem. For now, if you want to save plots with `.ipynb`, please use `%matplotlib inline`.

from .project_logger import logger
from .main import calHash
import pandas as pd
import numpy as np
import os


def getStats(dataDf, experiments, title=''):
    """
    Usage: meanDf, nquantDf, varDf, stdDf, semDf = getStats(lfqDf, experiments)
    If you want to exclude any data, do that before passing data in here.
    """
    # calculate hash for parameters
    ha = calHash(dataDf, experiments)  # skipping title for now

    os.makedirs('dataTables', exist_ok=True)

    meanTbFile = f'dataTables/mean_{title}_{ha}.tsv'.replace('__', '_')
    nquantTbFile = f'dataTables/nquant_{title}_{ha}.tsv'.replace('__', '_')
    varTbFile = f'dataTables/var_{title}_{ha}.tsv'.replace('__', '_')
    stdTbFile = f'dataTables/std_{title}_{ha}.tsv'.replace('__', '_')
    semTbFile = f'dataTables/sem_{title}_{ha}.tsv'.replace('__', '_')

    outputFiles = [meanTbFile, nquantTbFile, varTbFile, stdTbFile, semTbFile]

    if all(os.path.isfile(f) for f in outputFiles):
        outputDfs = [pd.read_csv(f, sep='\t', index_col=0) for f in outputFiles]
        [logger.info(f'Read basic stats from {f}') for f in outputFiles]
        return outputDfs
    conditions = list(experiments.keys())
    logger.info('Calculate stats with ddof at least 1 (ignoring the conditions with only one quantification).\n')
    resDf = pd.DataFrame(index=dataDf.index, columns=conditions)
    meanDf = resDf.copy()
    stdDf = resDf.copy()
    nquantDf = resDf.copy()
    semDf = resDf.copy()
    varDf = resDf.copy()

    for c in conditions:
        data = dataDf.loc[:, experiments[c]]

        mean = np.nanmean(data, axis=1)
        n = (~np.isnan(data)).sum(axis=1)
        var = np.nanvar(data, axis=1, ddof=1)
        std = np.sqrt(var)
        sem = std / np.sqrt(n)
        for df, x in zip(
            [meanDf, nquantDf, varDf, stdDf, semDf],
            [mean, n, var, std, sem]
        ):
            df.loc[:, c] = x

    for df, f in zip([meanDf, nquantDf, varDf, stdDf, semDf], outputFiles):
        df.to_csv(f, sep='\t')
        logger.info(f'Write basic stats data to {f}.')

    return meanDf, nquantDf, varDf, stdDf, semDf
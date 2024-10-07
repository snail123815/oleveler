import os
from .project_logger import logger
from hashlib import md5
import pandas as pd
import numpy as np

def genComparisonResults(compResultFile, comparisons):
    """
    allComparisons - dict of {'comp1':DF, 'comp2':DF,...}

    DF.columns = ['log2FC', 'SE', 'pvalue', 'adj.pvalue', 'ImputationPercentage']
    for MSstats result
    DF.columns = ['baseMean', 'log2FC', 'SE', 'pvalue', 'adj.pvalue']
    for DESeq2 result
    """
    compResultDf = pd.read_csv(compResultFile, sep='\t', index_col=0)

    if 'ImputationPercentage' in compResultDf.columns:
        # MSstats output
        # "Label"	"log2FC"	"SE"	"Tvalue"	"DF"	"pvalue"	"adj.pvalue"	"issue"	"MissingPercentage"	"ImputationPercentage"
        targetCols = ['log2FC', 'SE', 'pvalue', 'adj.pvalue', 'ImputationPercentage']
    elif 'baseMean' in compResultDf.columns:
        # DESeq2 result
        # "Label" is added while combining all data
        # "baseMean"	"log2FoldChange"	"lfcSE"	"pvalue"	"padj"
        # for shrinked data, note the column "lfcSE" is actually "posterior SD"
        # OR
        # "baseMean"	"log2FoldChange"	"lfcSE"	"stat"	"pvalue"	"padj"
        # for non-shrinked data
        targetCols = ['log2FC', 'SE', 'pvalue', 'adj.pvalue', 'baseMean']
    else:
        raise ValueError("Do not know result type, make sure 'ImputationPercentage' or 'baseMean' in input data columns")

    allCompResults = {}
    for c in comparisons:
        compData = compResultDf[compResultDf.Label == c]
        compData = compData.loc[:, targetCols]
        allCompResults[c] = compData

    return allCompResults


def _checkExistingCompResult(compExcel, sourceDf, compResultFileBase):
    # compResultFile => path, see if the file exists
    comparisons = pd.read_excel(compExcel, index_col=0).T.to_dict()
    allCompResults = None
    with open(compExcel, 'rb') as f:
        cur_compExcelHash = md5(f.read()).hexdigest()
    cur_sDfHash = md5(sourceDf.to_json().encode()).hexdigest()
    compResultFile, ext = os.path.splitext(compResultFileBase)
    compResultFile += f'_{cur_compExcelHash[:6]}_{cur_sDfHash[:6]}{ext}'
    if os.path.isfile(compResultFile):
        logger.info(f'Current {compExcel} hash = {cur_compExcelHash}.')
        logger.info(f'Current input table hash = {cur_sDfHash}.')
        logger.info(f'They should match previous result {compResultFile}.')
        # check if current comparisons.xlsx match previous
        allCompResults = genComparisonResults(compResultFile, comparisons)
    return allCompResults, comparisons, compResultFile


def getSig(compDf, tFc=1.5, tPv=0.05):
    """return compDf with only significant ids

    Note if you need significant ids in only one experiment, do not pass whole compDf

    Args:
        compDf ([type]): [description]
        tFc (float, optional): [description]. Defaults to 1.5.
        tPv (float, optional): [description]. Defaults to 0.05.

    Returns:
        compDf: filtered
    """
    f = compDf['log2FC'] > np.log2(tFc)
    f = f | (compDf['log2FC'] < -np.log2(tFc))
    f = f & (compDf['adj.pvalue'] < tPv)
    return compDf[f]


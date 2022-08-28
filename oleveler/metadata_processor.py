from .project_logger import logger
from .safe_annotations import safeCol
from collections import OrderedDict
import pandas as pd


def loadMeta(path, toRemove=[]):
    '''
    Usage: 
        metaDf, conditions, experiments = loadMeta('Annotation.csv')

    Read metadata file. Now using the criteria of MSstats:
    Columns of the metadata file (csv):
    Raw.file,Condition,BioReplicate,Experiment
    Thes columns can be used in MSstats analysis
    This will be further translated to a DESeq2 required file on the fly.

    The difference between metadata files used by MSstats and DESeq2 is that
    DESeq2 accepts multiple factor table while MSstats only accepts one factor: Condition.
    This makes meta data transformation easier from MSstats to DESeq2. 


    Return metaDf, conditions, experiments

    metaDf => pd.DataFrame
        indexs = names of raw files, DO NOT include extension
        cols = Raw.file, Condition, BioReplicate, Experiment
            eg.
                Condition = strain+condition
                BioReplicate = 1,1,1,2,2,2,3,3,3
                Experiment = strain+condition+repnumber (single experiment)
    conditions => list
    experiments => dict(cond1:[exp1, exp2, exp3], cond2:[exp4, exp5, exp6]...)
    '''
    logger.info(f'####### Load metadata #######')
    logger.info(f'Metadata path: {path}')
    metaDf = pd.read_csv(path, index_col=0)
    assert not metaDf.index[0].endswith('.raw'), f'Please remove ".raw" from table {path}'
    metaDf.loc[:,"Experiment"] = safeCol(metaDf['Experiment'])
    metaDf.loc[:,"Condition"] = safeCol(metaDf['Condition'])
    toRemove = safeCol(toRemove)
    metaDf = metaDf[~metaDf['Experiment'].isin(toRemove)]
    conditions = list(set(metaDf.Condition.to_list()))
    conditions.sort()
    logger.info('All conditions: ' + ', '.join(conditions))

    experiments = OrderedDict()
    for c in conditions:
        exps = metaDf.Experiment[metaDf.Condition == c].to_list()
        exps.sort()
        experiments[c] = exps
    logger.info(f'####### END Load metadata #######\n')

    return metaDf, conditions, experiments


def removeExperiments(metaDf, experiments, toRemove, dataDf=None):
    # TODO update
    # ??
    """
    Usage:
        newMetaDf, newConditions, newExperiments, newDataDf = removeExperiments(metaDf, experiments, ['condA_rep_1', 'condB_rep_3'], dataDf)
    toRemove should be a list of experiment names (columns of the data table) to remove
    Provided list should have exact match.
    experiments => dict(cond1:[exp1, exp2, exp3], cond2:[exp4, exp5, exp6]...)
    """
    logger.info(f'Remove {toRemove} from experiments')
    newExps = OrderedDict()
    newMetaDf = pd.DataFrame()
    for c in experiments:
        exps = [e for e in experiments[c] if e not in toRemove]
        if len(exps) > 0:
            newExps[c] = exps
            newMetaDf = pd.concat((newMetaDf, metaDf.loc[metaDf['Experiment'].isin(exps), :]), axis=0)

    if isinstance(dataDf, type(None)):
        newDataDf = None
    else:
        newDataDf = dataDf.loc[:, dataDf.columns.isin(toRemove)]
    newConditions = list(newExps.keys())

    return newMetaDf, newConditions, newExps, newDataDf


def removeIds(dataDf, toRemove):
    """Usage:
        newDf = removeIds(dataDf, ['idx1', 'idx2', ...])

    Args:
        dataDf (pd.DataFrame): input data table
        toRemove (list): list of ids to remove. Will consider partial match.

    Returns:
        newDf: removed df
    """
    logger.info(f'Remove genes/proteins in this list {toRemove}')
    newDf = dataDf.copy()
    newIdx = [i for i in newDf.index if all(s not in i for s in toRemove)]
    removed = newDf.index.difference(newIdx)
    logger.info(f'Actual ids removed: {list(removed)}')
    newDf = newDf.loc[newIdx, :]
    return newDf

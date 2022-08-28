from oleveler import logger, calHash
import os
import pickle
import pandas as pd
import numpy as np
from oleveler.safe_annotations import safeCol



def loadMQLfqData(dataPath, minLfq=3, toRemove=[]):
    '''
    For proteomics data, process MaxQuant out put so that the result can be feed to
    DESeq2. Also good for showing the origional data.
    Import MaxQuant LFQ result. Filter out the identifications with less than 3; 
    divide all numbers by 1000 to make it fits int32 for R. 
    '''
    logger.info("####### MaxQuant LFQ data input #######")
    scriptPath = './'
    dataTablesPath = os.path.join(scriptPath, 'dataTables')
    os.makedirs(dataTablesPath, exist_ok=True)

    proteinGroupFilePath = os.path.join(
        dataPath,
        'proteinGroups.txt'
    )
    proteinGroupsDf = pd.read_csv(
        proteinGroupFilePath,
        sep='\t',
        header=0,
        index_col='id',
        # to use read in all data without warning.
        # The table is quite complicated, and has
        # mixed types for some columns if pandas
        # infer data type of each column.
        low_memory=False
    )
    ha = calHash(proteinGroupFilePath, minLfq, toRemove)

    lfqTableOut = os.path.join(
        dataTablesPath,
        f'MQ_LFQ_table_{ha}.tsv'
    )
    lfqPickleOut = os.path.join(
        dataTablesPath,
        f'MQ_LFQ_table_{ha}_with_id.df.pickle'
    )

    if os.path.isfile(lfqTableOut) and os.path.isfile(lfqPickleOut):
        logger.info('Found processed LFQ data in program folder.')
        logger.info(lfqPickleOut)
        logger.info(lfqTableOut)
        with open(lfqPickleOut, 'rb') as f:
            lfqDf, id2group = pickle.load(f)
    else:
        logger.info('LFQ data not found in program folder, start processing of proteinGroups.txt file.')
        # MaxQuant data processing
        logger.info(proteinGroupFilePath)
        logger.info('Remove "CON__" and "REV__" data')
        # remove 'CON__', 'REV__'
        proteinGroupsDf = proteinGroupsDf[
            ~proteinGroupsDf['Protein IDs'].str.contains('CON__') &
            ~proteinGroupsDf['Protein IDs'].str.contains('REV__')
        ]
        # get id from the table
        id2group = proteinGroupsDf['Protein IDs']
        proteinGroupsDf = proteinGroupsDf.set_index('Protein IDs')
        proteinGroupsDf.index.name = "Protein_IDs"
        # get column names of LFQ data
        lfqCols = proteinGroupsDf.columns[
            proteinGroupsDf.columns.to_series().str.contains('LFQ intensity ')
        ]
        # Remove columns except LFQ
        logger.info('Remove columns except LFQ')
        lfqDf = proteinGroupsDf[lfqCols]
        # Remove 'LFQ intensity ' from column names
        lfqDf.columns = safeCol(lfqDf.columns.to_series().str.replace('LFQ intensity ', ''))
        lfqDf = processMQLFQ(lfqDf, minLfq, toRemove)
        # Write out table
        logger.info(f'Write out table {lfqTableOut}')
        lfqDf.to_csv(lfqTableOut, sep='\t')
        with open(lfqPickleOut, 'wb') as f:
            pickle.dump([lfqDf, id2group], f)

    try:
        evidenceFilePath = os.path.join(dataPath, 'evidence.txt')
        ev = pd.read_csv(evidenceFilePath, sep='\t', usecols=['Raw file', 'Experiment'])
        rawFiles = ev['Raw file'].unique()
        logger.info('Raw files are:')
        logger.info(', '.join(rawFiles))
        experiments = safeCol(ev['Experiment'].unique())
        assert all(col in experiments for col in lfqDf.columns)
    except FileNotFoundError:
        logger.warning('evidence.txt file not found, MSstats will not work.')
    except AssertionError as e:
        logger.error('Protein groups file and evidence file do not match')
        logger.info('Experiments in evidence.txt')
        logger.info(experiments)
        logger.info('Experiments in ProteinGroups.txt')
        logger.info(lfqDf.columns)
        raise e

    logger.info("####### END MaxQuant LFQ data input #######\n")
    return lfqDf, id2group


def processMQLFQ(lfqDf, minLfq=3, toRemove=[]):
    lfqDf = lfqDf.copy()
    for exp in toRemove:
        lfqDf.drop(exp, axis=1, inplace=True)
        logger.info(f'Experiment {exp} is removed from LFQ result of MaxQuant.')
    nProtGroupsAll = lfqDf.shape[0]
    logger.info(f'{nProtGroupsAll} protein groups in MQ output, including LFQ zeros.')
    # Reduce numbers by 1000
    logger.info('Reduce numbers by 1000')
    lfqDf = lfqDf.replace(np.nan, 0)
    lfqDf = lfqDf / 1000
    # Make all numbers as int type for DESeq2
    logger.info('Make all numbers as int type for DESeq2')
    lfqDf = lfqDf.astype('int')
    # Replace 0 with NA
    logger.info('Replace 0 with NA')
    lfqDf = lfqDf.replace(0, np.nan)
    # remove all zeros and < minLfq quantifications
    logger.info(f'Remove < {minLfq} quantifications')
    lfqDf = lfqDf[(~lfqDf.isna()).sum(axis=1) >= minLfq]
    nProtGroups = lfqDf.shape[0]
    logger.info(f'Removed {nProtGroupsAll - nProtGroups} protein groups, now {nProtGroups}.')
    return lfqDf

from oleveler import logger, calHash
from pathlib import Path
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
    logger.info(f'{nProtGroupsAll} rows in MQ output, including LFQ zeros.')
    # Reduce numbers by 1000
    logger.info('Reduce numbers by 1000')
    lfqDf = lfqDf.replace(np.nan, 0)
    lfqDf = lfqDf / 1000
    # Make all numbers as int type for DESeq2
    logger.info('Round up all numbers for DESeq2')
    lfqDf = lfqDf.round()
    # Replace 0 with NA
    logger.info('Replace 0 with NA')
    lfqDf = lfqDf.replace(0, np.nan)
    # remove all zeros and < minLfq quantifications
    logger.info(f'Remove < {minLfq} quantifications')
    lfqDf = lfqDf[(~lfqDf.isna()).sum(axis=1) >= minLfq]
    nProtGroups = lfqDf.shape[0]
    logger.info(f'Removed {nProtGroupsAll - nProtGroups} rows, now {nProtGroups}.')
    return lfqDf


# Replacement of
# loadMQLfqData(dataPath, minLfq=3, toRemove=[]):
# processMQLFQ(lfqDf, minLfq=3, toRemove=[]):
# NOTE: the id2group variable is NOT returned!!!!

def processAnyLFQ(lfqDf, minLfq=3, toRemove=[]):
    lfqDf = lfqDf.copy()
    for exp in toRemove:
        lfqDf.drop(exp, axis=1, inplace=True)
        logger.info(f"Experiment {exp} is removed from table.")
    nProtGroupsAll = lfqDf.shape[0]
    logger.info(
        f"{nProtGroupsAll} protein groups from input table, including zeros."
    )
    lfqDf = lfqDf.replace(np.nan, 0)

    # Reduce data range if max value exceed int32 max value in R
    max_value = lfqDf.to_numpy().max()
    if max_value >= 2147483647:
        reducing_factor = np.ceil(max_value / 2147483647)
        logger.info(
            "Max value of input data is larger than int32 "
            f"can store, reduced by factor of {reducing_factor}"
        )
        lfqDf = lfqDf / reducing_factor

    # Replace 0 with NA
    logger.info("Replace 0 with NA")
    lfqDf = lfqDf.replace(0, np.nan)
    # remove all zeros and < minLfq quantifications
    logger.info(
        f"Remove protein groups that have < {minLfq} quantifications"
        " across ALL samples."
    )
    lfqDf = lfqDf[(~lfqDf.isna()).sum(axis=1) >= minLfq]
    nProtGroups = lfqDf.shape[0]
    logger.info(
        f"Removed {nProtGroupsAll - nProtGroups} protein groups,"
        f" now {nProtGroups}."
    )
    logger.info(
        "Make all numbers as int type for DESeq2, (nan will be 0 again)."
    )
    lfqDf = lfqDf.replace(np.nan, 0)
    lfqDf = lfqDf.astype("int")
    return lfqDf


def loadAnyTable(dataPath, minLfq=3, toRemove=[]) -> pd.DataFrame:
    """
    For proteomics data, process output so that the result can be feed to
    DESeq2. Also good for showing the origional data.
    Filter out the identifications with less than 3;
    Check all data if they fit int32 for R, if not, reduce.

    Data format:
    1. First row is experiment names
    2. First column is protein group names
    3. Except first row and column, all other parts are data.
    """
    logger.info("####### Any data input #######")
    dataTablesPath = Path("./dataTables")
    dataTablesPath.mkdir(exist_ok=True)

    ha = calHash(dataPath, minLfq, toRemove)

    proteinGroupsDf = pd.read_csv(
        dataPath,
        sep="\t",
        header=0,
        index_col=0,
        # to use read in all data without warning.
        # The table is quite complicated, and has
        # mixed types for some columns if pandas
        # infer data type of each column.
        low_memory=False,
    )

    lfqTableOut = dataTablesPath / f"Any_PG_table_{ha}.tsv"
    lfqPickleOut = dataTablesPath / f"Any_PG_table_{ha}_with_id.df.pickle"

    if lfqTableOut.exists() and lfqPickleOut.exists():
        logger.info("Found processed LFQ data in program folder.")
        logger.info(lfqPickleOut)
        logger.info(lfqTableOut)
        with open(lfqPickleOut, "rb") as f:
            lfqDf = pickle.load(f)
    else:
        logger.info("Processing protein groups table.")
        # MaxQuant data processing
        logger.info(dataPath)
        # get id from the table
        proteinGroupsDf.index.name = "Protein_IDs"
        # get column names of LFQ data
        lfqDf = processAnyLFQ(proteinGroupsDf, minLfq, toRemove)
        # Write out table
        logger.info(f"Write out table {lfqTableOut}")
        lfqDf.to_csv(lfqTableOut, sep="\t")
        with open(lfqPickleOut, "wb") as f:
            pickle.dump(lfqDf, f)

    logger.info("####### END Any data input #######\n")
    return lfqDf

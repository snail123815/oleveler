import os
import pickle
import logging
import sys
from multiprocessing import Process
from tempfile import NamedTemporaryFile
from hashlib import md5
from datetime import datetime
from collections import OrderedDict
from time import sleep  # may be used in the notebook

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

from BCBio import GFF

import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.rinterface_lib.embedded import RRuntimeError

from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import matplotlib.ticker as mtick

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression as PLS
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope

logger = logging.getLogger()
logging.getLogger('matplotlib.font_manager').disabled = True
if len(logger.handlers) == 0:
    logFormatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] - %(message)s'
    )
    logfhandler = logging.FileHandler('dataProcessing.log')
    logfhandler.setLevel(logging.INFO)
    logfhandler.setFormatter(logFormatter)
    logshandler = logging.StreamHandler(stream=sys.stdout)
    logshandler.setLevel(logging.DEBUG)
    logshandler.setFormatter(logFormatter)
    logger.addHandler(logfhandler)
    logger.addHandler(logshandler)
    logger.setLevel(logging.DEBUG)

# ==================== LOAD DATA =======================


def loadMQLfqData(dataPath, minLfq=3):
    '''
    For proteomics data, process MaxQuant out put so that the result can be feed to
    DESeq2. Also good for showing the origional data.
    Import MaxQuant LFQ result. Filter out the identifications with less than 3; 
    divide all numbers by 1000 to make it fits int32 for R. 
    '''
    logger.info("####### MaxQuant LFQ data input #######")
    scriptPath = './'
    dataTablesPath = os.path.join(scriptPath, 'dataTables')
    if not os.path.isdir(dataTablesPath):
        os.makedirs(dataTablesPath)

    proteinGroupFilePath = os.path.join(
        dataPath,
        'proteinGroups.txt'
    )
    with open(proteinGroupFilePath, 'rb') as pg:
        ha = md5(pg.read()).hexdigest()[:6]

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
        lfqDf.columns = lfqDf.columns.to_series().str.replace('LFQ intensity ', '')
        nProtGroupsAll = lfqDf.shape[0]
        logger.info(f'{nProtGroupsAll} protein groups in MQ output, including LFQ zeros.')
        # Reduce numbers by 1000
        logger.info('Reduce numbers by 1000')
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
        # Add id column to lfqDf
        logger.info('Add id column to lfqDf ')
        # Write out table
        logger.info(f'Write out table {lfqTableOut}')
        lfqDf.to_csv(lfqTableOut, sep='\t')
        with open(lfqPickleOut, 'wb') as f:
            pickle.dump([lfqDf, id2group], f)
    logger.info("####### END MaxQuant LFQ data input #######\n")
    return lfqDf, id2group


def loadMeta(path):
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
    #metaDf.index = [i.replace('.raw','') for i in metaDf.index]
    conditions = list(set(metaDf.Condition.to_list()))
    conditions.sort()

    experiments = OrderedDict()
    for c in conditions:
        exps = metaDf.Experiment[metaDf.Condition == c].to_list()
        exps.sort()
        experiments[c] = exps
    logger.info(f'####### END Load metadata #######\n')

    return metaDf, conditions, experiments


def loadCountTable(tablePath, cleanId=[]):
    """Usage:
        loadCountTable('novogeneCounts/readcount.xls', cleanId=['gene-'])
    Read count table and remove some SUBstring from the ID. eg. 'gene-abc' when 'gene-' in
    cleanId list will become 'abc' in the output table.

    Args:
        tablePath (str): path to the count table
        cleanId (list, optional): SUB string to remove from every id. Defaults to [].

    Returns:
        pd.DataFrame: count table, usually with experiments as column, genes as index.
    """
    logger.info(f'Load count table from file {tablePath}')
    ct = pd.read_csv(tablePath, sep='\t', index_col=0, header=0)
    idx = ct.index
    for s in cleanId:
        idx = idx.map(lambda x: x.replace(s, ''))
    ct.index = idx
    return ct


def gatherCountTable(countsPath):
    """Usage: 
        ct = gatherCountTable('selfAlign/gene_counts')
    Prepare count table from seperated count files produced by [featureCounts] (subread)

    Args:
        countsPath (str): path to the folder containing featureCounts outputs

    Returns:
        pd.DataFrame: count table
    """
    logger.info(f'Raw count table gathered from {countsPath}')
    ct = pd.DataFrame()
    flist = os.listdir(countsPath)
    flist.sort()
    for f in flist:
        fn, ext = os.path.splitext(f)
        if ext.lower() in ['.summary', '.log', '.gff']:
            continue
        elif ext == '.txt':
            fp = os.path.join(countsPath, f)
            c = pd.read_csv(fp, sep='\t', comment='#',
                            header=0, index_col=0).iloc[:, 5]
            ct[fn] = c
    return ct


def _parseGff(gff, tags, getTypes=['rRNA']):
    """Usage:
    geneLengths, targetTypes = _parseGff('filePath', ['locus_tag'], getTypes=['rRNA'])

    Do two things: 
    1. get the length of each gene. 
    2. get lists of target gene types listed in getTypes

    Args:
        gff (str): path to gff/gtf file
        tags (list): a list of qualifier names that can be parsed as gene ID.
        getTypes (list, optional): An extra list will generate with the length pdSeries.
                                   Defaults to ['rRNA'].

    Returns:
        geneLengths (pd.Series): gene ID as index
        targetTypes: list of lists. eg. getTypes=['rRNA','sRNA'] then 
                     targetTypes = [['rRNA_ID1', 'rRNA_ID2',...], 
                                    ['sRNA_ID1', 'sRNA_ID2',...]]
    """

    logger.info(f'Gathering information from annotation file {gff}, will take some time...')
    targetTypes = [[] for t in getTypes]
    geneLengths = pd.Series(dtype=int)
    if isinstance(tags, str):
        tags = [tags]
    with open(gff, 'r') as f:
        for rec in GFF.parse(f):
            for feat in rec.features:
                geneId = ''
                if feat.type != 'gene':
                    continue
                for tag in tags:
                    if tag in feat.qualifiers:
                        geneId = feat.qualifiers[tag][0]
                        break
                assert geneId != '', f'Gene name not found with tag "{tags}", feature:\n{feat}'
                geneLen = len(feat)
                geneLengths[geneId] = geneLen
                for i, t in enumerate(getTypes):
                    if t in feat.qualifiers['gene_biotype']:
                        targetTypes[i].append(geneId)
    return geneLengths, targetTypes


def _parseTableForGeneLength(tableFile, lengthColParsingKeys=['length'],
                             typeCol='type', getTypes=['rRNA']):
    """Usage:
        geneLengths, targetTypes = \
            _parseTableForGeneLength('tableFilePath', lengthColParsingKeys=['length'],
                                     typeCol='gene_biotype', getTypes=['rRNA'])

    Do two things: 
    1. get the length of each gene. 
    2. get lists of target gene types listed in getTypes

    Note for ['.xls', '.tsv', '.txt'] I assume sep='\\t', for ['.csv'] I assume sep=','
    Note I assume there is header, and geneid should be in the first columns as index

    Args:
        tableFile (str): path to table file
        lengthColParsingKeys (list, optional): column names for gene length. Will try 
                                               sequencially follow this list. Will stop when
                                               one matching is found.
                                               Defaults to ['length'].
                                               Note: partial match is enough
        typeCol (str, optional): column name from which you get the IDs that belong to 
                                 target types. Defaults to 'type'. 
                                 Note: partial match is enough
        getTypes (list, optional): An extra list will generate with the length pdSeries.
                                   Defaults to ['rRNA'].

    Returns:
        geneLengths (pd.Series): gene ID as index
        targetTypes: list of lists. eg. getTypes=['rRNA','sRNA'] then 
                     targetTypes = [['rRNA_ID1', 'rRNA_ID2',...], 
                                    ['sRNA_ID1', 'sRNA_ID2',...]]
    """
    ext = os.path.splitext(tableFile)[1]
    if ext in ['.xls', '.tsv', '.txt']:
        sep = '\t'
    elif ext in ['.csv']:
        sep = ','
    try:
        tb = pd.read_csv(tableFile, sep=sep, index_col=0, header=0, comment='#')
    except UnicodeDecodeError:  # if file extension .xls is indeed excel file
        tb = pd.read_excel(tableFile)
    lk = lengthColParsingKeys.copy()
    k1 = lk.pop(0)
    lc = [l for l in tb.columns if k1 in l]
    while len(lc) > 1 and len(lk) > 0:
        k = lk.pop(0)
        lc = [l for l in lc if k in l]
    if len(lc) > 1:
        logger.warning(fr'''Found multiple length columns in {tableFile} with keys {lengthColParsingKeys}:
                        {lc}
                        Using the first one. Consider adding more key word to the selection.''')
    lc = lc[0]
    geneLengths = tb[lc]
    geneLengths.name = None  # good for concating
    targetTypes = [[] for t in getTypes]
    typeCols = [c for c in tb.columns if typeCol in c]
    if len(typeCols) > 0:
        for tpc in typeCols:
            for i, t in enumerate(getTypes):
                targetTypes[i].extend(tb.index[tb[tpc] == t].to_list())
    return geneLengths, targetTypes


def _cleanCountTable(countTableDf, infoFiles,
                     tagsForGeneName='locus_tag',
                     lengthColParsingKeys=['length', 'gene'], typeCol='type',
                     removerRNA=True, removeIDsubstrings=[], removeIDcontains=[]):
    """
    for gff files
    infoFiles = ['a.gff', 'b.gff'] # should also support .gtf format
    tagsForGeneName = [['tag1for_a', 'tag2for_a'], ['tag1for_b', 'tag2for_b']] # can also be str

    for other tables
    infoFiles = ['*.txt', '*.csv', '*.tsv', '*.xlsx']
    lengthColParsingKeys=['length', 'gene'], typeCol='type', 

    """
    # Define targets
    rrnas = []
    geneLengths = pd.Series(dtype=int)

    if isinstance(infoFiles, str):
        infoFiles = [infoFiles]
    if isinstance(tagsForGeneName, str):
        tagsForGeneName = [tagsForGeneName]
    if isinstance(lengthColParsingKeys, str):
        lengthColParsingKeys = [lengthColParsingKeys]

    gffFiles = []
    otherFiles = []
    for f in infoFiles:
        ext = os.path.splitext(f)[1]
        if ext in ['.gff', '.gtf']:
            gffFiles.append(f)
        else:
            otherFiles.append(f)
    files = gffFiles + otherFiles
    assert len(files) >= len(tagsForGeneName)
    tagsForGeneName += [[] for i in range(len(files)-len(tagsForGeneName))]

    for i, (f, tags) in enumerate(zip(files, tagsForGeneName)):
        if i < len(gffFiles):
            gls, rs = _parseGff(f, tags, getTypes=['rRNA'])
        else:
            gls, rs = _parseTableForGeneLength(f, lengthColParsingKeys=lengthColParsingKeys,
                                               typeCol=typeCol, getTypes=['rRNA'])
        rrnas.extend(rs[0])
        geneLengths = pd.concat((geneLengths, gls))

    for substr in removeIDsubstrings:
        geneLengths.index = geneLengths.index.map(lambda s: s.replace(substr, ''))
    if removerRNA and len(rrnas) > 0:
        logger.info(f'Found rRNA gene info in {infoFiles}, {len(rrnas)} in total, first several being:\n{rrnas[:min(5, len(rrnas))]}')
        logger.info(f'If these ids are present in the count table, they will be removed.')
        newct = countTableDf.loc[countTableDf.index.map(lambda x: x not in rrnas), :]
        logger.info(f'Removed {countTableDf.shape[0] - newct.shape[0]} rRNA genes')
    else:
        newct = countTableDf.copy()
    for s in removeIDcontains:
        l = newct.shape[0]
        newct = newct.loc[~newct.index.str.contains(s), :]
        logger.info(f'Removed {l - newct.shape[0]} genes with "{s}" in there id')
    return newct, geneLengths


def _TPM(ct, geneLengths):
    """transcripts per million

    Args:
        ct (pd.DataFrame): cleaned count table
        geneLengths (pd.Series): lengths of every gene

    Returns:
        pd.DataFrame: calculated TPM
    """
    geneLengths = geneLengths[ct.index]
    # convert to array for easy calculation
    ctArray = ct.to_numpy()
    glArray = geneLengths.to_numpy()
    # normalize for gene length
    res = ctArray/glArray[:, np.newaxis]
    # normalize for sequencing depth
    res = res/res.sum(axis=0)[np.newaxis, :]
    # per million
    res = res*1000000
    res = pd.DataFrame(res, index=ct.index, columns=ct.columns)
    logger.info('TPM calculated')
    return res


def calculateTPM(countTableDf, infoFiles,
                 tagsForGeneName='locus_tag',
                 lengthColParsingKeys=['length', 'gene'], typeCol='type',
                 removerRNA=True, removeIDsubstrings=[], removeIDcontains=[]):
    """Major function for RNA seq TPM table calculation.

    # ct = prepareCountTable("selfAlign/gene_counts")
    # calculateTPM(ct, 'selfAlign/gene_counts/GCF_000203835.1_ASM20383v1_genomic.gff', 
    #              tagsForGeneName='locus_tag', removerRNA=True, removeIDcontains=['SCP'])

    # ct = loadCountTable('novogeneCounts/readcount.xls', cleanId=['gene-'])
    # calculateTPM(ct, 
    #              ['novogeneCounts/gene.xls', 'novogeneCounts/sRNA_length.xls'], 
    #              removerRNA=True, 
    #              removeIDsubstrings=['gene-'], removeIDcontains=['sRNA', 'Novel'])

    Args:
        countTableDf (pd.DataFrame): pd.DataFrame of raw counts
        infoFiles (str): path to annotation table or GFF/GTF files. Used in removal rRNA etc.
        tagsForGeneName (str, optional): Feature qualifier name for parsing gene name.
                                         Gene names should present in count table index. 
                                         Used only when GFF/GTF files provided.
                                         Defaults to 'locus_tag'.
        lengthColParsingKeys (list, optional): The column names for parsing gene length,
                                               first first, partial match. 
                                               Defaults to ['length', 'gene'].
        typeCol (str, optional): column name for parsing gene type (eg. to see if this gene is 
                                 rRNA or not). Partial match. Defaults to 'type'.
        removerRNA (bool, optional): Whether the result table include rRNA. Usually rRNA was
                                     removed from sample before cDNA synthesis, the quantity
                                     of removed rRNA varias and should be discarded.
                                     Defaults to True.
        removeIDsubstrings (list, optional): Any substrings in this list will be checked if any ID
                                             contains them, if so, the substring will be removed from
                                             the ID name. eg. to remove 'gene-' from 'gene-dasR' will
                                             result in 'dasR' as the ID of this gene. 
                                             Defaults to [].
        removeIDcontains (list, optional): IDs that contain any substrings in this list will be
                                           removed. Good to remove some unnecessary genes or sRNAs.
                                           Defaults to [].

    Returns:
        pd.DataFrame: TPM calculated
    """
    with NamedTemporaryFile() as f:
        countTableDf.to_csv(f, sep='\t')
        for p in [infoFiles, tagsForGeneName, lengthColParsingKeys, typeCol, 
            str(removerRNA), removeIDsubstrings, removeIDcontains]:
            f.write(''.join(p).encode())
        f.seek(0)
        paraHash = md5(f.read()).hexdigest()
    if not os.path.isdir('dataTables'):
        os.makedirs('dataTables')
    tpmTablePath = f'dataTables/TPM_Table_{paraHash}.tsv'
    if os.path.isfile(tpmTablePath):
        tpmDf = pd.read_csv(tpmTablePath, sep='\t', index_col=0)
        logger.info(f'TPM data read from {tpmTablePath}.')
    else:
        newct, geneLengths = _cleanCountTable(countTableDf, infoFiles,
                                            tagsForGeneName,
                                            lengthColParsingKeys, typeCol,
                                            removerRNA, removeIDsubstrings, removeIDcontains)
        tpmDf = _TPM(newct, geneLengths)
        tpmDf.to_csv(tpmTablePath, sep='\t')
        logger.info(f'TPM data write to table {tpmTablePath}.')
    return tpmDf


# ====================== DATAP ROCESSING ======================

# Basic stats

def removeExperiments(dataDf, metaDf, experiments, toRemove):
    """
    Usage:
        newDataDf, newMetaDf, newConditions, newExperiments = removeExperiments(dataDf, metaDf, experiments, ['condA_rep_1', 'condB_rep_3'])
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
            newMetaDf = pd.concat((newMetaDf, metaDf.loc[metaDf['Experiment'].isin(exps),:]), axis=0)
    
    newDataDf = dataDf.loc[:, dataDf.columns.isin(toRemove)]
    newConditions = list(newExps.keys())
    
    return newDataDf, newMetaDf, newConditions, newExps


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


def getStats(dataDf, experiments):
    """
    Usage: meanDf, nquantDf, varDf, stdDf, semDf = getStats(lfqDf, experiments)
    If you want to exclude any data, do that before passing data in here.
    """
    # calculate hash for parameters
    with NamedTemporaryFile() as f:
        dataDf.to_csv(f.name, sep='\t')
        f.write(''.join(experiments.keys()).encode())
        f.write(''.join(''.join(experiments[k]) for k in experiments).encode())
        f.seek(0)
        tbHash = md5(f.read()).hexdigest()[:6] 
        
    meanTbFile = f'dataTables/mean_{tbHash}.tsv'
    nquantTbFile = f'dataTables/nquant_{tbHash}.tsv'
    varTbFile = f'dataTables/var_{tbHash}.tsv'
    stdTbFile = f'dataTables/std_{tbHash}.tsv'
    semTbFile = f'dataTables/sem_{tbHash}.tsv'

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


# DESeq2 VST transformation
def genInfoDfDESeq2(dataDf, metaDf):
    '''
    dataDf.column should have same elements as metaDf.Experiment
    '''
    infoDf = metaDf.Condition.to_frame()
    infoDf.index = metaDf.Experiment
    infoDf = infoDf.reindex(index=dataDf.columns)
    tempInfoFile = NamedTemporaryFile()
    infoDf.to_csv(tempInfoFile.name, sep='\t')
    features = infoDf.columns
    return tempInfoFile, features


def writeDataForDESeq2(dataDf):
    '''
    Will replace np.nan with zero for R import
    A named temporary file will be used rather than writing a real data.
    tempCleanDataFile, outForR = writeDataForR(theDf)'''
    # fill with zeros and convert to int for DESeq2 input
    feedToRDf = dataDf.replace(np.nan, 0).round().astype('int')
    tempCleanDataFile = NamedTemporaryFile()
    feedToRDf.to_csv(tempCleanDataFile.name, sep='\t')
    return tempCleanDataFile


def deseq2Process(
    dataDf,
    metaDf,
    ref=None,
    deOnly=False,
    designCol=0,
):
    """
    Will return vstDf by default (deOnly = False).
    If you want to do difference analysis, pass getDE=True
    """
    pathTransformed = 'dataTables/transformed'
    if not os.path.isdir(pathTransformed):
        os.makedirs(pathTransformed)
    # calculate hash
    with NamedTemporaryFile() as f:
        dataDf.to_csv(f.name)
        with NamedTemporaryFile() as g:
            metaDf.to_csv(g.name)
            mhb = md5(g.read()).digest()
        f.write(mhb)
        f.write(str(ref).encode())
        f.write(str(deOnly).encode())
        f.write(str(designCol).encode())
        f.seek(0)
        ha = md5(f.read()).hexdigest()[:6]

    pathVst = os.path.join(pathTransformed, f'vst_{ha}.tsv')

    if os.path.isfile(pathVst) and not deOnly:
        logger.info(f'Read VST transformed data from {pathVst}.')
        vstDf = pd.read_csv(pathVst, sep='\t', header=0, index_col=0)
        return vstDf

    rpackages.importr("DESeq2")

    logger.info('####### Process data using DESeq2 #######')
    tempDataFile = writeDataForDESeq2(dataDf)
    tempInfoFile, features = genInfoDfDESeq2(dataDf, metaDf)
    logger.info('Fitting data using DESeq2...')
    idxName = dataDf.index.name
    robjects.r(
        f"""
        coldata = read.csv('{tempInfoFile.name}', sep='\t', row.names = 1, header = TRUE)
        cts = as.matrix(read.csv('{tempDataFile.name}',sep='\t', row.names = '{idxName}'))
        print(all(rownames(coldata) == colnames(cts)))
        """
    )

    design = features[designCol]

    logger.info(f'Set factor for "design": "{design}"')
    robjects.r(
        f"""
        coldata${design} = factor(coldata${design})
        """
    )
    if isinstance(ref, type(None)):
        ref = metaDf.loc[:, design].sort_values()[0]
        logger.info(f'Set reference {design} (default) to {ref}')
    else:
        logger.info(f'Set reference {design} to {ref}')
    robjects.r(
        f"""
        coldata${design} = relevel(coldata${design}, ref=as.character("{ref}"))
        """
    )
    #robjects.r(
    #    f"""
    #    print(coldata${design})
    #    """
    #)
    robjects.r(
        f"""
        dds = DESeqDataSetFromMatrix(
            countData = cts,
            colData = coldata,
            design = ~{design}
        )
        dds <- DESeq(
            dds, test = "Wald", fitType = "parametric", sfType = "ratio",
            betaPrior = FALSE, quiet = FALSE, useT = FALSE,
            minmu = 0.5, parallel = FALSE
        )
        """
    )
    tempInfoFile.close()
    tempDataFile.close()
    logger.info('END DESeq2 data init with current design:')
    if deOnly:
        return design
    else:
        logger.info('Performing VST')
        robjects.r(
            f"""
            vsd = vst(dds, blind=FALSE)
            write.table(assay(vsd), file='{pathVst}', sep='\t', col.names=NA)
            """
        )
        logger.info(pathVst)

        logger.info('####### END Processing data using DESeq2 #######')

        vstDf = pd.read_csv(pathVst, sep='\t', header=0, index_col=0)
        return vstDf


# MSstats proposition


def clearlogfiles(msstatsLogPath='dataTables/transformed/MSstats_log/'):
    if not os.path.isdir(msstatsLogPath):
        os.makedirs(msstatsLogPath)
    # move log file to log folder
    for f in os.listdir():
        if f.startswith('MSstats_') and f.endswith('.log'):
            os.rename(f, os.path.join(msstatsLogPath, f))


def prepareMSstats(mqDataPath, annotationPath):
    """
    This will be reused in calculating differences if has not done in current session.
    """

    logger.info('####### MSstats import and proposition #######')

    logger.info('Import MSstats')
    rpackages.importr("MSstats")
    pgPath = os.path.join(mqDataPath, "proteinGroups.txt")
    evPath = os.path.join(mqDataPath, 'evidence.txt')
    logger.info('Read MaxQuant data')
    robjects.r(
        f"""
        proteinGroups <- read.table(
            "{pgPath}",
            sep="\t",
            header=TRUE
        )
        infile <- read.table(
            "{evPath}",
            sep="\t",
            header=TRUE
        )
        annot <- read.csv("{annotationPath}", header=TRUE)
        quant <- MaxQtoMSstatsFormat(
            evidence = infile, annotation = annot,
            proteinGroups = proteinGroups,
            removeProtein_with1Piptide = TRUE
        )
        """
    )

    robjects.r(
        f"""
        maxquant.proposed <- dataProcess(
            quant,
            normalization='equalizeMedians',
            summaryMethod="TMP",
            censoredInt='NA', # MaxQuant must
            MBimpute=TRUE,
            maxQuantileforCensored=0.999
        )
        """
    )
    logger.info('####### END MSstats import and proposition #######')
    clearlogfiles()


def processMSstats(mqDataPath, annotationPath):
    # calculate Hash
    pgPath = os.path.join(mqDataPath, "proteinGroups.txt")
    evPath = os.path.join(mqDataPath, 'evidence.txt')
    haFile = NamedTemporaryFile()
    with open(pgPath, 'rb') as f:
        haFile.write(md5(f.read()).digest())
    with open(evPath, 'rb') as f:
        haFile.write(md5(f.read()).digest())
    with open(annotationPath, 'rb') as f:
        haFile.write(md5(f.read()).digest())
    haFile.seek(0)
    ha = md5(haFile.read()).hexdigest()[:6]
    haFile.close()

    qcplotPath = f'dataTables/transformed/msstats_proposed_{ha}_QCPlot.pdf'
    proposedDataPath = f'dataTables/transformed/msstats_proposed_{ha}.tsv'
    logDataPath = f'dataTables/transformed/msstats_proposed_{ha}_logTable.tsv'

    # Calculate quantitation table
    if os.path.isfile(proposedDataPath):
        quantTable = pd.read_csv(proposedDataPath, sep='\t')
    else:
        prepareMSstats(mqDataPath, annotationPath)
        logger.info('Write proposed data to table')
        logger.info(proposedDataPath)
        robjects.r(
            f"""
            write.table(
                maxquant.proposed$ProteinLevelData,
                "{proposedDataPath}",
                sep='\t',
                row.names=FALSE
            )
            """
        )
        logger.info('Generate QC Plot for normalised data.')
        # produce a QCplot
        robjects.r(
            f"""
            dataProcessPlots(
                data=maxquant.proposed,
                type='QCplot',
                which.Protein="allonly"
            )
            """
        )
        # move the QCPlot to transformed data folder
        if os.path.isfile('QCPlot.pdf') and not os.path.isfile(qcplotPath):
            os.rename('QCPlot.pdf', qcplotPath)
        clearlogfiles()
        quantTable = pd.read_csv(proposedDataPath, sep='\t')

    # Convert the quantification data from MSstats to a expression table
    if os.path.isfile(logDataPath):
        logger.info(f'Read MSstats proposed expression table')
        logDf = pd.read_csv(logDataPath, sep='\t', index_col=0)
    else:
        logger.info('Transform MSstats proposed data into data table')
        logger.info(logDataPath)
        pgIds = pd.unique(quantTable.Protein)  # protein groups
        pgIds.sort()
        runIds = pd.unique(quantTable.originalRUN)
        runIds.sort()
    #     run2exp = metaDf.Experiment.to_dict()
        metaDf, _, _ = loadMeta(annotationPath)
        run2exp = metaDf.Experiment.to_dict()
        logDf = pd.DataFrame(index=pgIds)
        for run in runIds:
            experiment = run2exp[run]
            data = quantTable.loc[:, ['Protein', 'LogIntensities']][quantTable.originalRUN == run]
            data = data.set_index('Protein', drop=True)
            logDf[experiment] = data  # Some protein will be missing but will be filled
        logDf = logDf.reindex(sorted(logDf.columns), axis=1)
        logDf.to_csv(logDataPath, sep='\t')

    return quantTable, logDf


def msstatsComp(comparisons, mqDataPath, annotationPath, compResultFile):

    _, conditions, _ = loadMeta(annotationPath)
    # the file should be based on the comparisons column
    # columns = [id,exp,con]
    oneRowMatrixs = []
    compBindLines = ['comparisons <- rbind(', ]
    compNameLines = ['rownames(comparisons) <- c(', ]
    for c in comparisons:
        nexp = conditions.index(comparisons[c]['exp'])
        nctr = conditions.index(comparisons[c]['ctr'])
        matRow = []
        for i in range(len(conditions)):
            n = 0
            if i == nexp:
                n = 1
            elif i == nctr:
                n = -1
            matRow.append(f'{n},')
        matRow[-1] = matRow[-1][:-1]
        matRow = f'comparison.{c} <- matrix(c({"".join(matRow)}),nrow=1)'
        oneRowMatrixs.append(matRow)
        compBindLines.append(f'    comparison.{c},')
        compNameLines.append(f'    "{c}",')

    compBindLines[-1] = compBindLines[-1][:-1]
    compNameLines[-1] = compNameLines[-1][:-1]
    compBindLines.append(')')
    compNameLines.append(')')

    robjects.r('\n'.join(oneRowMatrixs))
    robjects.r('\n'.join(compBindLines))
    robjects.r('\n'.join(compNameLines))
    try:
        robjects.r(
            """
            colnames(comparisons) <- levels(maxquant.proposed$ProteinLevelData$GROUP)
            """
        )
    except RRuntimeError:
        if any((isinstance(mqDataPath, type(None)), isinstance(annotationPath, type(None)))):
            logger.error("msstatsComp() failed, please passin mqDataPath and annotationPath")
            raise ValueError
        logger.info('Re-generate maxquant data in R for new comparison')
        prepareMSstats(mqDataPath, annotationPath)
        robjects.r(
            """
            colnames(comparisons) <- levels(maxquant.proposed$ProteinLevelData$GROUP)
            """
        )
    logger.info(f'Calculating comparisons...')
#     robjects.r(
#     """
#     print('group')
#     print(head(maxquant.proposed$ProteinLevelData$Group))
#     print('data')
#     print(head(maxquant.proposed$ProteinLevelData))
#     print(levels(maxquant.proposed$ProteinLevelData$GROUP))
#     print(comparisons)
#     """)
    robjects.r(
        f"""
        maxquant.comparisons <- groupComparison(
            contrast.matrix = comparisons,
            data = maxquant.proposed
        )
        write.table(
            maxquant.comparisons$ComparisonResult,
            '{compResultFile}',
            sep='\t',
            row.names=FALSE
        )
        """
    )
    # keep following columns, and change name if required
    targetColsOri = ['Label', 'log2FC', 'SE', 'pvalue', 'adj.pvalue', 'issue', 'MissingPercentage', 'ImputationPercentage']
    targetCols = ['Label', 'log2FC', 'SE', 'pvalue', 'adj.pvalue', 'issue', 'MissingPercentage', 'ImputationPercentage']
    compResultDf = pd.read_csv(compResultFile, sep='\t', index_col=0)
    compResultDf = compResultDf.loc[:, targetColsOri]
    compResultDf.columns = targetCols
    compResultDf.to_csv(compResultFile, sep='\t')
    return compResultFile


def _deseq2Comp(a, b, name, extra, subResFile):
    robjects.r(
        f"""
        #print(resultsNames(dds))
        res <- {a}(dds, {b}="{name}", parallel=TRUE{extra})
        
        write.table(as.data.frame(res), col.names=NA, sep='\t',
                    file="{subResFile.name}")
        """
    )


def deseq2Comp(comparisons, countTable, annotationPath, compResultFile, lfcShrink=True, timeout=100):
    """Warning: Do not make too many comparisions in the comparisons file, it will cause R to consume too
    much memory and stop.

    Args:
        comparisons ([type]): [description]
        countTable ([type]): [description]
        annotationPath ([type]): [description]
        compResultFile ([type]): [description]
        lfcShrink (bool, optional): [description]. Defaults to True.
        timeout (int, optional): Note this will cause error if the processing in R indeed cost much time. Defaults to 300.

    Returns:
        [type]: [description]
    """
    metaDf, _, _ = loadMeta(annotationPath)
    allCompResults = OrderedDict()
    previousCtr = ''
    previousDesign = ''
    # DESeq2 dds is control condition sensitive.
    # Sort the comparisons based on the control conditions to reduce time consumption
    # in creating dds in DESeq2 with same control.
    sortedComps = sorted(comparisons, key=lambda x: comparisons[x]['ctr'])
    nDeseqRuns = len(set(comparisons[c]['ctr'] for c in comparisons))
    if nDeseqRuns > 4:
        logger.warning(rf'''There are more than 5 different controls to run, note the program may crash.
        please restart python kernel (thus you also killed R kernel) and run from the first cell.
        If that don't work, then you need to reduce the number of controls from the comparisons.
        Or you can use a better computer with larget memory to try.''')
    for c in sortedComps:
        exp = comparisons[c]['exp']
        ctr = comparisons[c]['ctr']
        logger.info(f'Calculat comparisons {exp} vs. {ctr}.')
        if ctr == previousCtr:
            design = previousDesign
        else:
            design = deseq2Process(countTable, metaDf, ref=ctr, deOnly=True)
            previousDesign = design
            previousCtr = ctr
        name = f'{design}_{exp}_vs_{ctr}'
        subResFile = NamedTemporaryFile()
        if lfcShrink:
            a = 'lfcShrink'
            b = 'coef'
            extra = ', type="apeglm"'
            logger.info('Using lfcShrink of the log2fc, ' +
                        'note the column "lfcSE" in shrinked data is actually "posterior SD"')
        else:
            a = 'results'
            b = 'name'
            extra = ', alpha=0.05'
        # Code below within this loop is to prevent R from lose respond.
        # Sometime (more likely to happen with lfcShrink) the calculation is done but python
        # does not get the signal to proceed.
        # Once a time out is added, then the possiblity that R cannot finish on time have to
        # be considered. The solution is to increase given time in a while loop.
        # rpy works by running a Python process and an R process in parallel,
        # and exchange information between them.
        # It does not take into account that R calls are called in parallel using multiprocess.
        # So in practice, each of the python processes connects to the same R process.
        maxRetryTime = 30*60  # 30 min
        i = 0
        t = timeout
        while os.path.getsize(subResFile.name) == 0:
            if i > 0:
                logger.info(f'Failed to calculate comparison within {int(t/60)} min, retry.')
            p = Process(target=_deseq2Comp, args=(a, b, name, extra, subResFile))
            subResFile.seek(0)  # Not necessary, r should write the file in R process
            p.start()
            p.join(timeout=t)
            t += timeout * max(0, i-1)  # retry once with the same time out, then increase this time.
            if t > maxRetryTime:
                break
            i += 1
        try:
            subResDf = pd.read_csv(subResFile.name, sep='\t', index_col=0)
            allCompResults[c] = subResDf
        except EmptyDataError:
            logger.warning('Failed calculating comparison with ' +
                           f'{i} trials, max run time per trial is {maxRetryTime/60} min. ' +
                           f'The comparison {exp} vs {ctr} with name {c} ignored.')
        subResFile.close()
    # Write to compResultFile
    # Generate a dataframe that store all reulsts in one table like MSstats
    allCompDf = pd.DataFrame()
    for c in comparisons:
        subResDf = allCompResults[c].copy()
        cols = ['Label'] + subResDf.columns.to_list()
        subResDf["Label"] = c
        subResDf.index.name = 'ID'
        subResDf = subResDf[cols]
        # Move  index to columns
        subResDf = subResDf.reset_index()
        allCompDf = pd.concat([allCompDf, subResDf], axis=0, ignore_index=True)
    allCompDf = allCompDf.set_index('ID', drop=True)
    targetColsOri = ['Label', 'log2FoldChange', 'lfcSE', 'pvalue', 'padj',       'baseMean']
    targetCols = ['Label', 'log2FC',         'SE',    'pvalue', 'adj.pvalue', 'baseMean']
    allCompDf = allCompDf.loc[:, targetColsOri]
    allCompDf.columns = targetCols
    allCompDf.to_csv(compResultFile, sep='\t')
    return compResultFile


def genComparisonResults(compResultFile, comparisons):

    #     newCompDf = pd.Dataframe()
    #     newCompTablePath = 'dataTables/transformed/msstats_proposed_comparisonResult_readable.tsv'
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
        raise

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
    with NamedTemporaryFile() as f:
        sourceDf.to_csv(f.name, sep='\t')
        cur_sDfHash = md5(f.read()).hexdigest()
    compResultFile, ext = os.path.splitext(compResultFileBase)
    compResultFile += f'_{cur_compExcelHash[:6]}_{cur_sDfHash[:6]}{ext}'
    if os.path.isfile(compResultFile):
        logger.info(f'Current {compExcel} hash = {cur_compExcelHash}.')
        logger.info(f'Current input table hash = {cur_sDfHash}.')
        logger.info(f'They should match previous result {compResultFile}.')
        # check if current comparisons.xlsx match previous
        allCompResults = genComparisonResults(compResultFile, comparisons)
    return allCompResults, comparisons, compResultFile


def makeCompMatrixMsstats(compExcel, mqDataPath, annotationPath):
    '''

    Return msstatsComparisonResultDf, comparisons

        msstatsComparisonResultDf - DataFrame output of ComparisonResult
        comparisons - dict of {'comp1':{'exp':'a', 'ctr':'b'},
                               'comp2':{'exp':'c', 'ctr':'d'},
                               ...}

    # Convert to data tables for each comparison.
    #
    # > The result of the test for diffrential abundance is a table with
    # > columns Protein, Label (of the comparison), log2 fold change (log2FC), 
    # > standard error of the log2 fold change (SE), test statistic of the 
    # > Student test (Tvalue), degree of freedom of the Student test (DF), 
    # > raw p-values (pvalue), p-values adjusted among all the proteins in 
    # > the specific comparison using the approach by Benjamini and Hochberg 
    # > (adj.pvalue). The cutoff of the adjusted p-value corresponds to the 
    # > cutoff of the False Discovery Rate (Benjamini and Hochberg 1955). 
    # > The positive values of log2FC for Label=C2-C1 indicate evidence in 
    # > favor of C2 > C1 (i.e. proteins upregulated in C2), while the negative 
    # > values indicate evidence in favor of C2 < C1 (i.e. proteins 
    # > downregulated in C2), as compared to C1. The same model can be used 
    # > to perform several comparisons of conditions simultaneously in the 
    # > same protein.
    # >
    # > **NOTE** `issue` column shows if there is any issue for inference in 
    # > corresponding protein and comparison, for example, OneConditionMissing 
    # > or CompleteMissing. If one of condition for compariosn is completely 
    # > missing, it would flag with OneConditionMissiong with adj.pvalue=0 and 
    # > log2FC=Inf or -Inf even though pvalue=NA. For example, if you want to 
    # > compare ‘Condition A - Condition B’, but condition B has complete missing, 
    # > log2FC=Inf and adj.pvalue=0. SE, Tvalue, and pvalue will be NA. if you 
    # > want to compare ‘Condition A - Condition B’, but condition A has complete 
    # > missing, then log2FC=-Inf and adj.pvalue=0. But, please be careful for 
    # > using this log2FC and adj.pvalue.
    '''
    # Process input see if anything has changed.
    # If not, read from existing result if have.
    compResultFileBase = 'dataTables/transformed/msstats_proposed_comparisonResult.tsv'

    lfqDf, _ = loadMQLfqData(mqDataPath) # only used for check if result exists, to avoid extensive R calls
    allCompResults, comparisons, compResultFile = \
        _checkExistingCompResult(compExcel, lfqDf, compResultFileBase)
    if not isinstance(allCompResults, type(None)):
        return allCompResults, comparisons

    logger.info(f'Parse {compExcel}, generate comparison matrix for MSstats.')
    compResultFile = msstatsComp(comparisons, mqDataPath, annotationPath, compResultFile)
    allCompResults = genComparisonResults(compResultFile, comparisons)
    logger.info(f'Done calculation, dump data in {compResultFile}\n')
    clearlogfiles()
    return allCompResults, comparisons


def makeCompMatrixDeseq2(compExcel, countTable, annotationPath, shrink=None):
    '''
    shrink only affects lfc values, not pvalues

    # Convert to data tables for each comparison.
    #
    # > The result of the test for diffrential abundance is a table with
    # > columns Protein, Label (of the comparison), log2 fold change (log2FC), 
    # > standard error of the log2 fold change (SE), test statistic of the 
    # > Student test (Tvalue), degree of freedom of the Student test (DF), 
    # > raw p-values (pvalue), p-values adjusted among all the proteins in 
    # > the specific comparison using the approach by Benjamini and Hochberg 
    # > (adj.pvalue). The cutoff of the adjusted p-value corresponds to the 
    # > cutoff of the False Discovery Rate (Benjamini and Hochberg 1955). 
    # > The positive values of log2FC for Label=C2-C1 indicate evidence in 
    # > favor of C2 > C1 (i.e. proteins upregulated in C2), while the negative 
    # > values indicate evidence in favor of C2 < C1 (i.e. proteins 
    # > downregulated in C2), as compared to C1. The same model can be used 
    # > to perform several comparisons of conditions simultaneously in the 
    # > same protein.
    # >
    # > **NOTE** `issue` column shows if there is any issue for inference in 
    # > corresponding protein and comparison, for example, OneConditionMissing 
    # > or CompleteMissing. If one of condition for compariosn is completely 
    # > missing, it would flag with OneConditionMissiong with adj.pvalue=0 and 
    # > log2FC=Inf or -Inf even though pvalue=NA. For example, if you want to 
    # > compare ‘Condition A - Condition B’, but condition B has complete missing, 
    # > log2FC=Inf and adj.pvalue=0. SE, Tvalue, and pvalue will be NA. if you 
    # > want to compare ‘Condition A - Condition B’, but condition A has complete 
    # > missing, then log2FC=-Inf and adj.pvalue=0. But, please be careful for 
    # > using this log2FC and adj.pvalue.
    '''
    # Process input see if anything has changed.
    # If not, read from existing result if have.
    if shrink == 'lfcShrink':
        compResultFileBase = 'dataTables/transformed/deseq2_comparisonResult_lfcShrink.tsv'
    elif isinstance(shrink, type(None)):
        compResultFileBase = 'dataTables/transformed/deseq2_comparisonResult.tsv'
    else:
        raise ValueError(f'shrink should be in [None, "lfcShrink"], but now it is {shrink}')

    allCompResults, comparisons, compResultFile = \
        _checkExistingCompResult(compExcel, countTable, compResultFileBase)
    if not isinstance(allCompResults, type(None)):
        return allCompResults, comparisons

    logger.info(f'Parse {compExcel}, generate comparison matrix using DESeq2.')
    compResultFile = deseq2Comp(comparisons, countTable, annotationPath, compResultFile,
                                lfcShrink=(True if shrink == 'lfcShrink' else False))
    allCompResults = genComparisonResults(compResultFile, comparisons)
    logger.info(f'Done calculation, dump data in {compResultFile}\n')
    return allCompResults, comparisons




def square_subplots(fig, axs):
    """Make a figure square"""
    try:
        ax1 = axs.ravel()[0]
    except:
        ax1 = axs
    rows, cols = ax1.get_subplotspec().get_gridspec().get_geometry()
    l = fig.subplotpars.left
    r = fig.subplotpars.right
    t = fig.subplotpars.top
    b = fig.subplotpars.bottom
    wspace = fig.subplotpars.wspace
    hspace = fig.subplotpars.hspace
    figw, figh = fig.get_size_inches()
    axw = figw * (r - l) / (cols + (cols - 1) * wspace)
    axh = figh * (t - b) / (rows + (rows - 1) * hspace)
    axs = min(axw, axh)
    w = (1 - axs / figw * (cols + (cols - 1) * wspace)) / 2.0
    h = (1 - axs / figh * (rows + (rows - 1) * hspace)) / 2.0
    fig.subplots_adjust(bottom=h, top=1 - h, left=w, right=1 - w)


def _extendRange(a, scale=1.1):
    """make a range(x,y) larger.
    newx < x < y < newy

    Args:
        a (tuple): x, y
        scale (float, optional): fold to increase. Defaults to 1.1.

    Returns:
        (newx, newy): extended range
    """
    x, y = a
    r = y - x
    assert r > 0
    m = x + r/2
    r = r * scale
    x = m - r/2
    y = m + r/2
    return (x, y)


def genColour(idx, sep='_', cmap='turbo'):
    """
    Generate set colours based on the prefix of each elements in idx.

    eg. passing idx = ['a_1', 'a_2', 'b_1', 'b_2'] will give four colours,
    with the first two alike each other, last two alike each other, vast
    differences between the first two and last two. ie. two group of colours.

    This function assume it can distingush group names using 'sep' to 
    split each element of idx. Else it will simply generate sequential
    colours.
    """
    idx = list(idx)
    idx.sort()
    cmap = cm.get_cmap(cmap)
    groups = OrderedDict()
    colours = []
    for i in idx:
        sp = i.split(sep)
        if len(sp) > 1:
            g = sep.join(sp[:-1])
        else:
            g = sp[0]
        if g not in groups:
            groups[g] = [i]
        else:
            groups[g].append(i)
    gap = 1/(len(groups) + 1)
    majorColours = np.arange(0+gap, 1, gap)
    for g, mc in zip(groups, majorColours):
        n = len(groups[g])
        sgap = 0.5*gap/n
        colours.extend(np.linspace(mc, mc+0.5*gap-sgap, n))
    return [cmap(i) for i in colours]



def genPlsClasses(idx, sep='_'):
    """generate classes for use in PLS based on the prefix of each element
    in idx.

    Args:
        idx (list/pd.Series): list which group can be distingushed by their
                              prefix
        sep (str, optional): seperator. Defaults to '_'.

    Returns:
        classes: generated classes info for use in PLS. A list of numbers
                 with each number represents one class/group.
    """
    idx = list(idx)
    groups = OrderedDict()
    classes = []
    for i in idx:
        sp = i.split(sep)
        if len(sp) > 1:
            g = sep.join(sp[:-1])
        else:
            g = sp[0]
        if g not in groups:
            groups[g] = [i]
        else:
            groups[g].append(i)
    for i, g in enumerate(groups):
        classes.extend([i]*len(groups[g]))
    return classes


def _mixColors(colors):
    """Mix rgb colors (darker), take a list of colors and mix them"""
    added = []
    for c in zip(*colors):
        added.append(sum(c))
    mixed = [i / len(colors) for i in added]
    return tuple(mixed)


def _rotateAxis(alpha, df):
    """df.shape = (n, 2)"""
    rotated = df.copy()
    cos = np.cos(alpha / 180 * np.pi)
    sin = np.sin(alpha / 180 * np.pi)
    for i, (a, b) in df.iterrows():
        rotated.loc[i, :] = (a * cos + b * sin), (b * cos - a * sin)
    return rotated


def plotCorr(df, cols=None, method='pearson', fontsize=6, figsize=(6, 5),
             vmin=0.9, vmax=1, colormap='Greens'):
    """Make correlation plot for the data. Each column treated as one element
    in the correlation plot.

    Args:
        df (pd.DataFrame): dataframe to deal with
        cols (list, optional): columns to use. Defaults to None.
        method (str, optional): will pass to pd.DataFrame().corr(method=method). Defaults to 'pearson'.
        fontsize (int, optional): ticklabels font size. Defaults to 6.
        figsize (tuple, optional): figure size. Defaults to (6, 5).
        vmin (float, optional): minimum correlation to plot. Defaults to 0.9.
        vmax (int, optional): maximum correlation to plot. Defaults to 1.
        colormap (str, optional): colormap name used in matplotlib. Defaults to 'Greens'.
    """
    if not isinstance(cols, type(None)):
        df = df.loc[:, cols]
    else:
        cols = df.columns
    corrDf = df.corr(method=method)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    pcm = ax.pcolormesh(corrDf, vmin=vmin, vmax=vmax, cmap=colormap)
    ticks = np.arange(0.5, df.shape[1]+0.5, 1)
    ax.set_xticks(ticks)
    ax.set_xticklabels(cols, fontsize=fontsize, rotation=90)
    ax.set_yticks(ticks)
    ax.set_yticklabels(cols, fontsize=fontsize)
    ax.get_xaxis().set_ticks_position('top')
    ax.set_ylim(ax.get_ylim()[::-1])
    ax.set_title('Correlation Matrix')
    fig.colorbar(pcm, ax=ax)
    plt.tight_layout()
    plt.show()


# PCA and PLS

def getNtopVar(df, ntop=300):
    """produce a new df containing only the top n rows based on the variance

    Args:
        df (pd.DataFrame): data table
        ntop (int, optional): number of elements to keep. Defaults to 300.
    
    Return:
        newDf: containing only ntop elements (rows)
    """
    if ntop >= df.shape[0]:
        logger.info(f'ntop {ntop}>= number of ids {df.shape[0]}')
        return df
    vs = df.var(axis=1)
    vs = vs.sort_values(ascending=False)
    newDf = df.loc[vs.iloc[:ntop].index, :]
    return newDf
    

def doPCA(df, ntop=None):
    """do PCA on df, return pca table and PcaClass

    Args:
        df (pd.DataFrame): data to process
        ntop (int, optional): only take the top n variance elements for computing. Defaults to None.

    Returns:
        pcaDf, PcaClass
    """
    PcaClass = PCA()
    if ntop != None:
        df = getNtopVar(df, ntop=ntop)
    try:
        pcaDF = pd.DataFrame(
            PcaClass.fit_transform(df.transpose()),
            index=df.columns,
            columns=list(f"PC{i+1}" for i in range(PcaClass.n_components_)),
        )
    except ValueError:
        logger.info("Input of doPCA(df) contains unsupported values, please use transformed data.")
        raise
    return pcaDF, PcaClass



def doPLS(df, classes=None, ntop=None, n_components=2):
    """do supervised PLS analysis on df, return pls data, PlsClass, and R2 of the fit

    Args:
        df (pd.DataFrame): data to process
        classes (list, optional): list of classes/groups of each column. So len(classes)
                                  should equal to len(df.columns). Defaults to None, will
                                  treat every column as seperate groups.
        ntop (int, optional): Only take the top n variance elements for computing. Defaults to None.
                              Defaults to None.
        n_components (int, optional): Number of components that PLS will use. Defaults to 2.
    
    Return:
        plsDf, PlsClass, r2
    """
    PlsClass = PLS(n_components=n_components)
    if isinstance(classes, type(None)):
        classes = np.arange(0, df.shape[1], 1).reshape(-1, 1)
    else:
        classes = np.array(classes)
    if ntop != None:
        df = getNtopVar(df, ntop=ntop)
#     print(df.T.shape)
#     print(classes.shape)
    x_scores, y_scores = PlsClass.fit_transform(df.T, classes)
    plsDf = pd.DataFrame(x_scores, index=df.columns,
                         columns=[f"PC{i+1}" for i in range(n_components)],)
    r2 = PlsClass.score(df.T, classes)
#     print(PlsClass.x_loadings_.shape)
    logger.info(f"PLS: Coefficient of determination R^2: {r2}")
    return plsDf, PlsClass, r2

# doPLS(vstDf, genClasses(vstDf.columns))


def plotPrincipleAnalysis(df, cols=None, note=None, ntop=None, figsize=(6, 5),
                          title="", colourSet=None, rotation=0, sciLabel=False,
                          analysisType='PCA', plsClasses=None, square=True):
    """ Use normalised data for plotting, if not, data cannot contain nan valuse"""
    plotName = f'{analysisType} plot - {title}'
    logger.info(f'Plotting {plotName}')
    plt.close(plotName)
    if cols != None:
        df = df.loc[:, cols]
    else:
        cols = df.columns
    if analysisType == "PCA":
        paDf, PaClass = doPCA(df, ntop=ntop)
    elif analysisType == 'PLS':
        if isinstance(plsClasses, type(None)):
            plsClasses = genPlsClasses(df.columns)
            logger.info('Using default seperator for genClases() - "_"')
            logger.info(plsClasses)
        paDf, PaClass, r2 = doPLS(df, plsClasses, ntop=ntop)
    else:
        raise
#     print(paDf.head())
    directory = f'Plots/{analysisType}/'
    name = f'{analysisType}_{title}'
    if not os.path.isdir(directory):
        os.makedirs(directory)
    name += ('_' if title != '' else '')
    name += "_".join(cols[:min(len(cols), 2)])
    if len(cols) > 2:
        name += '...'
    filePath = os.path.join(directory, name)

    dataPlot = paDf.loc[:, ["PC1", "PC2"]]
    if rotation != 0:
        filePath += f'_rotated{rotation}_'
        dataPlot = _rotateAxis(rotation, dataPlot)
    timeStr = datetime.now().strftime("%Y%m%d%H%M%S")
    filePath += timeStr

    experiments = dataPlot.index
    colours = []
    labels = []
    legends = []
#     print(experiments)

    # markerTypes = cycle(["o", "s", "^", "d", "p", "P", "*"])
    for i, col in enumerate(cols):
        labels.append(col)
        if colourSet == None:
            colour = f"C{i}"
        else:
            colour = colourSet[i]
        colours.append(colour)
        legends.append(Line2D([0], [0], marker='o', color=colour,
                       markersize=8, markeredgewidth=0.5,
                       markeredgecolor='k',
                       linewidth=0, label=col))

    # start plotting
    fig, ax = plt.subplots(1, 1, figsize=figsize, num=plotName)
    sc = ax.scatter(dataPlot.PC1, dataPlot.PC2,
                    c=colours, s=65, marker="o",
                    zorder=3,
                    # zorder is to move the dots to front, or they will be covered by grid
                    edgecolors="k", linewidths=0.5)
    ax.grid(color="whitesmoke", linewidth=1)
    ax.legend(handles=legends, title="Experiments",
              labelspacing=0.1, loc="upper left",
              bbox_to_anchor=(1.01, 1))

    # text on plot
    ntopText = f" ntop {ntop}" if ntop != None else ""
    axTitle = f"2 components {analysisType} for {name}{ntopText}\n"
    if analysisType == 'PCA':
        pc1Ratio, pc2Ratio = PaClass.explained_variance_ratio_[:2]
        ax.set_xlabel(f"PC1: {pc1Ratio:0.1%} variance")
        ax.set_ylabel(f"PC2: {pc2Ratio:0.1%} variance")
        axTitle += f"(total variance representation: {pc1Ratio + pc2Ratio:0.1%})"
    else:
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        axTitle += r"Coefficient of determination $R^2$: " + f'{r2:.3f}'
    axTitle += ("" if note == None else "\n" + note)
    ax.set_title(axTitle, fontsize=10)
    if sciLabel:
        ax.ticklabel_format(style="sci", scilimits=(0, 2), useMathText=True)

    # hovering data and functions
    ymin, ymax = _extendRange(ax.get_ylim(), 1.1)
    xmin, xmax = _extendRange(ax.get_xlim(), 1.1)
    lnv = ax.plot([0, 0], [ymin, ymax], color='black', linewidth=0.3)[0]
    lnh = ax.plot([xmin, xmax], [0, 0], color='black', linewidth=0.3)[0]
    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))
    lnv.set_linestyle('None')
    lnh.set_linestyle('None')
    annot = ax.annotate('', xy=(0, 0), xytext=(5, 5), textcoords="offset points")
    annot.set_visible(False)

    def hover(event):
        if event.inaxes == ax:
            lnv.set_data([event.xdata, event.xdata], [ymin, ymax])
            lnh.set_data([xmin, xmax], [event.ydata, event.ydata])
            lnv.set_linestyle('--')
            lnh.set_linestyle('--')
            lnv.set_visible(True)
            lnh.set_visible(True)
            # Test if on data points
            cont, ind = sc.contains(event)
            if cont:
                annot.xy = (event.xdata, event.ydata)
                annot.set_text('\n'.join([labels[n] for n in ind['ind']]))
                annot.set_visible(True)
            else:
                annot.set_visible(False)
        else:
            lnv.set_visible(False)
            lnh.set_visible(False)
    fig.canvas.mpl_connect("motion_notify_event", hover)
#     plt.tight_layout()
    if square:
        square_subplots(fig, ax)
    logger.info(f'PCA plot generated with path {filePath}\n')
    fig.savefig(f'{filePath}.svg')
    paDf.to_excel(f'{filePath}.xlsx')
    plt.show()
    return PaClass


def plotPCAExplaination(PcaClass, title=""):
    plotName = f'PCA explaination - {title}'
    logger.info(f'Plotting {plotName}')
    plt.close(plotName)
    explainedRatios = PcaClass.explained_variance_ratio_
    explainedRatios = [r for r in explainedRatios if r >= 0.01]
    _, ax = plt.subplots(1, 1, num=plotName)
    xaxis = np.arange(1, len(explainedRatios) + 1)
    ax.bar(xaxis, explainedRatios)
    ax.set_xticks(xaxis)
    for i, ratio in enumerate(explainedRatios):
        ax.text(i + 1, ratio, f"{ratio:.2%}",
                ha="center", va="bottom", size="xx-small")
#     ax.set_yticklabels([f"{y:.0%}" for y in ax.get_yticks()])
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_ylabel("Explained variance ratio")
    ax.set_xticklabels([f"PC{x}" for x in xaxis], size="x-small")
    ax.set_xlabel("Principal components")
    ax.set_title(
        f"Explained variance in different principal components"
    )
    plt.show()


def outlierAlgorithm(outliersFraction, drawOutliers):
    anomalyAlgorithms = [
        OneClassSVM(nu=outliersFraction, kernel="rbf", gamma="auto", coef0=0.01),
        IsolationForest(contamination=outliersFraction, random_state=100),
        LocalOutlierFactor(contamination=outliersFraction, n_neighbors=100),
        EllipticEnvelope(contamination=outliersFraction),
    ]
    return anomalyAlgorithms[drawOutliers]


def assignColor(colorIdx, condition, color):
    subidx = colorIdx.loc[condition].index
    subSeries = pd.Series([color] * len(subidx), index=subidx)
    return pd.concat(
        [colorIdx.loc[-condition], subSeries], axis=0, sort=False
    )


def plotPrincipleAnalysisLoading(df, cols=None, note=None, ntop=None,
                                 figsize=(6, 5), title="", colourSet=None, rotation=0,
                                 sciLabel=False, drawOutliers=False, outlierDimension='2D',
                                 outlierAlg=0, outliersFraction=0.05, square=True,
                                 analysisType='PCA', plsClasses=None):
    """
# plotPrincipleAnalysisLoading(vstDf, drawOutliers=True, outlierAlg=0, outliersFraction=0.05, title='VST',
#                              analysisType="PCA")
# plotPrincipleAnalysisLoading(vstDf, drawOutliers=True, outlierAlg=0, outliersFraction=0.05, title='VST',
#                              analysisType="PLS")
    Do normalization before pass DataFrame here 
    outliersFraction=0.05,  # precentage of outliers
    """
    plotName = f'Loading plot {analysisType} - {title}'
    logger.info(f'Plotting {plotName}')
    plt.close(plotName)

    if cols != None:
        df = df.loc[:, cols]
    else:
        cols = df.columns
    if analysisType == "PCA":
        paDf, PaClass = doPCA(df, ntop=ntop)
        dfComp = pd.DataFrame(PaClass.components_.transpose(),
                              index=df.index, columns=paDf.columns)
    elif analysisType == 'PLS':
        if isinstance(plsClasses, type(None)):
            plsClasses = genPlsClasses(df.columns)
            logger.info('Using default seperator for genClases() - "_"')
            logger.info(plsClasses)
        paDf, PaClass, r2 = doPLS(df, plsClasses, ntop=ntop)
        dfComp = pd.DataFrame(PaClass.x_loadings_,
                              index=df.index, columns=paDf.columns)
    else:
        raise

    directory = f'Plots/{analysisType}/'
    name = f'{analysisType}_{title}'
    if not os.path.isdir(directory):
        os.makedirs(directory)
    name += ('_' if title != '' else '')
    name += 'loading_'
    name += "_".join(cols[:min(len(cols), 2)])
    if len(cols) > 2:
        name += '...'
    filePath = os.path.join(directory, name)

    dfPlot = dfComp.loc[:, ["PC1", "PC2"]]
    if rotation != 0:
        dfPlot = _rotateAxis(rotation, dfPlot)
    # set colors
    baseColor = "C0"
    outliersCl2d = mcolors.to_rgb("C2")
    outliersClC1 = mcolors.to_rgb("C2")
    outliersClC2 = mcolors.to_rgb("C3")

    if drawOutliers:
        result2D = outlierAlgorithm(outliersFraction, outlierAlg).fit_predict(dfPlot)
        outliersFraction = outliersFraction / 2
        resultC1 = outlierAlgorithm(outliersFraction, outlierAlg).fit_predict(
            dfPlot.PC1.to_frame()
        )
        resultC2 = outlierAlgorithm(outliersFraction, outlierAlg).fit_predict(
            dfPlot.PC2.to_frame()
        )
    else:
        result2D = np.array([1] * len(dfPlot.index))
        resultC1 = result2D.copy()
        resultC2 = result2D.copy()

    outliers2d = pd.Series(result2D, index=dfPlot.index, name="outliers_2D")
    outliersC1 = pd.Series(resultC1, index=dfPlot.index, name="outliers_C1")
    outliersC2 = pd.Series(resultC2, index=dfPlot.index, name="outliers_C2")

    # Determine point size based on the maximum expression of a protein
    # Size set in scatter plot is the suface of the point
    pointSizeData = df.mean(axis=1)
    transformedSizeData = MinMaxScaler().fit_transform(
        pointSizeData.to_numpy().reshape(-1, 1)
    ).flatten()
    pointSizeData = pd.Series(transformedSizeData, index=pointSizeData.index)

    def calculateSize(x):
        # s of the scatter function is area
        return x * 100 + 1
    pointSize = calculateSize(pointSizeData)

    colorIdx = pd.Series([baseColor] * len(dfPlot.index), index=dfPlot.index)
    if drawOutliers:
        if outlierDimension == '2D':
            colorIdx = assignColor(colorIdx, (outliers2d == -1), outliersCl2d)
        else:
            colorIdx = assignColor(colorIdx, (outliersC1 == -1), outliersClC1)
            colorIdx = assignColor(colorIdx, (outliersC2 == -1), outliersClC2)
            colorIdx = assignColor(colorIdx,
                                   ((outliersC1 == -1) & (outliersC2 == -1)),
                                   _mixColors([outliersClC2, outliersClC1]))
        colorIdx = colorIdx.loc[dfPlot.index]

    # Start plotting
    fig, ax = plt.subplots(1, 1, figsize=figsize, num=plotName)
    sc = ax.scatter(dfPlot.PC1, dfPlot.PC2, alpha=0.7,
                    c=colorIdx, s=pointSize, linewidths=0)

    # text on plot
    ntopText = f" ntop {ntop}" if ntop != None else ""
    axTitle = f"2 components {analysisType} for {name}{ntopText}\n"
    if analysisType == 'PCA':
        pc1Ratio, pc2Ratio = PaClass.explained_variance_ratio_[:2]
        ax.set_xlabel(f"PC1: {pc1Ratio:0.1%} variance")
        ax.set_ylabel(f"PC2: {pc2Ratio:0.1%} variance")
        axTitle += f"(total variance representation: {pc1Ratio + pc2Ratio:0.1%})"
    else:
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        axTitle += r"Coefficient of determination $R^2$: " + f'{r2:.3f}'
    axTitle += ("" if note == None else "\n" + note)
    ax.set_title(axTitle, fontsize=10)
    if sciLabel:
        ax.ticklabel_format(style="sci", scilimits=(0, 2), useMathText=True)

    # hovering data and functions
    labels = dfPlot.index.to_list()
    ymin, ymax = _extendRange(ax.get_ylim(), 1.1)
    xmin, xmax = _extendRange(ax.get_xlim(), 1.1)
    lnv = ax.plot([0, 0], [ymin, ymax], color='black', linewidth=0.3)[0]
    lnh = ax.plot([xmin, xmax], [0, 0], color='black', linewidth=0.3)[0]
    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))
    lnv.set_linestyle('None')
    lnh.set_linestyle('None')
    annot = ax.annotate('', xy=(0, 0), xytext=(5, 5), textcoords="offset points")
    annot.set_visible(False)

    def hover(event):
        if event.inaxes == ax:
            lnv.set_data([event.xdata, event.xdata], [ymin, ymax])
            lnh.set_data([xmin, xmax], [event.ydata, event.ydata])
            lnv.set_linestyle('--')
            lnh.set_linestyle('--')
            lnv.set_visible(True)
            lnh.set_visible(True)
            # Test if on data points
            cont, ind = sc.contains(event)
            if cont:
                annot.xy = (event.xdata, event.ydata)
                annot.set_text('\n'.join([(labels[n]) for n in ind['ind']]))
                annot.set_visible(True)
            else:
                annot.set_visible(False)
        else:
            lnv.set_visible(False)
            lnh.set_visible(False)
    fig.canvas.mpl_connect("motion_notify_event", hover)

    title = f"Loading principal axes ({name})"
    ax.set_title(title)
    if square:
        square_subplots(fig, ax)

    fig.savefig(f'{filePath}.svg')
    dfPlot.to_excel(f'{filePath}.xlsx')

    plt.show()



def calculateVips(model):
    """calculat VIP (Variable Importance in Projection) for V-plot
    https://www.researchgate.net/post/How_can_I_compute_Variable_Importance_in_Projection_VIP_in_Partial_Least_Squares_PLS
    Solution by Keiron Teilo O'Shea """
    t = model.x_scores_
    w = model.x_weights_
    q = model.y_loadings_
    p, h = w.shape
    vips = np.zeros((p,))
    s = np.diag(np.matmul(np.matmul(np.matmul(t.T, t), q.T), q)).reshape(h, -1)
    total_s = np.sum(s)
    for i in range(p):
        weight = np.array(
            [(w[i, j] / np.linalg.norm(w[:, j])) ** 2 for j in range(h)]
        )
        vips[i] = np.sqrt(p * (np.matmul(s.T, weight)) / total_s)
    return vips


def plotPlsVplot(df, ntop=None, classes=None, cols=None, n_components=2,
                 outlierAlg=3, outliersFraction=0.05, square=True,
                 title=None, tClass=0, drawOutliers=True, figsize=(6, 5)):
    """ Do normalization before pass DataFrame here """
    plotName = f'PLS V plot - {title}'
    logger.info(f'Plotting {plotName}')
    plt.close(plotName)

    if isinstance(cols, type(None)):
        cols = df.columns
    else:
        df = df.loc[:, cols]
    if isinstance(classes, type(None)):
        classes = genPlsClasses(df.columns)
        logger.info('Using default seperator for genClases() - "_"')
        logger.info(classes)
    plsDf, PlsClass, r2 = doPLS(df, classes, ntop=ntop)

    dfPlot = pd.DataFrame(
        np.array((PlsClass.coef_[:, tClass], calculateVips(PlsClass))).T,
        index=df.index,
        columns=["coef", "VIP"],
    )

    directory = f'Plots/PLS/'
    name = f'PLS_Vplot_{title}'
    if not os.path.isdir(directory):
        os.makedirs(directory)
    name += ('_' if title != '' else '')
    name += "_".join(cols[:min(len(cols), 2)])
    if len(cols) > 2:
        name += '...'
    filePath = os.path.join(directory, name)

    # Size set in scatter plot is the suface of the point
    pointSizeData = df.mean(axis=1)
    transformedSizeData = MinMaxScaler().fit_transform(
        pointSizeData.to_numpy().reshape(-1, 1)
    ).flatten()
    pointSizeData = pd.Series(transformedSizeData, index=pointSizeData.index)

    def calculateSize(x):
        # s of the scatter function is area
        return x * 100 + 1
    pointSize = calculateSize(pointSizeData)

    baseColor = "C0"
    outliersClC1 = mcolors.to_rgb("C2")

    if drawOutliers:
        outliersFraction = outliersFraction / 2
        resultC1 = outlierAlgorithm(outliersFraction, outlierAlg).fit_predict(
            dfPlot.coef.to_frame()
        )
    else:
        result2D = np.array([1] * len(dfPlot.index))
        resultC1 = result2D.copy()

    outliersC1 = pd.Series(resultC1, index=dfPlot.index, name="outliers_C1")

    colorIdx = pd.Series([baseColor] * len(dfPlot.index), index=dfPlot.index)
    if drawOutliers:
        colorIdx = assignColor(colorIdx, (outliersC1 == -1), outliersClC1)
        colorIdx = colorIdx.loc[dfPlot.index]

    # start plotting
    fig, ax = plt.subplots(1, 1, figsize=figsize, num=plotName)
    sc = ax.scatter(
        dfPlot.iloc[:, 0],
        dfPlot.iloc[:, 1],
        alpha=0.7,
        c=colorIdx,
        s=pointSize,
        linewidths=0,
    )
    ax.set_xlabel("Correlation Coefficient")
    ax.set_ylabel("Variable Importance in Projection")

    xlims = PlsClass.coef_[:, 0].min(), PlsClass.coef_[:, 0].max()
    maxXlim = max(abs(xlims[0]), abs(xlims[1]))
    expendX = maxXlim * 0.1
    xlims = [-maxXlim-expendX, maxXlim+expendX]
    ax.set_xlim(xlims)

    # hovering data and functions
    labels = dfPlot.index.to_list()
    ymin, ymax = _extendRange(ax.get_ylim(), 1.1)
    xmin, xmax = _extendRange(ax.get_xlim(), 1.1)
    lnv = ax.plot([0, 0], [ymin, ymax], color='black', linewidth=0.3)[0]
    lnh = ax.plot([xmin, xmax], [0, 0], color='black', linewidth=0.3)[0]
    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))
    lnv.set_linestyle('None')
    lnh.set_linestyle('None')
    annot = ax.annotate('', xy=(0, 0), xytext=(5, 5), textcoords="offset points")
    annot.set_visible(False)

    def hover(event):
        if event.inaxes == ax:
            lnv.set_data([event.xdata, event.xdata], [ymin, ymax])
            lnh.set_data([xmin, xmax], [event.ydata, event.ydata])
            lnv.set_linestyle('--')
            lnh.set_linestyle('--')
            lnv.set_visible(True)
            lnh.set_visible(True)
            # Test if on data points
            cont, ind = sc.contains(event)
            if cont:
                annot.xy = (event.xdata, event.ydata)
                annot.set_text('\n'.join([(labels[n]) for n in ind['ind']]))
                annot.set_visible(True)
            else:
                annot.set_visible(False)
        else:
            lnv.set_visible(False)
            lnh.set_visible(False)
    fig.canvas.mpl_connect("motion_notify_event", hover)

    title = f"PLS V-plot ({title})"
    ax.set_title(title, fontsize=12)
    if square:
        square_subplots(fig, ax)
    fig.savefig(f'{filePath}.svg')
    dfPlot.to_excel(f'{filePath}.xlsx')
    plt.show()

# plotPlsVplot(vstDf, classes=plsClasses, title='VST')


# Volcano plot


def plotVolcano(compDf, quantSeries, figsize=(6, 5),
                square=True, lfcThresh=1, pThresh=0.05, title=''):
    """
# c = 'mu_wt20'
# cols = metaDf[
#     (metaDf.Condition==comparisons[c]['exp']) | \
#     (metaDf.Condition==comparisons[c]['ctr'])
# ].Experiment.values
# plotVolcano(
#     deseq2CompResultsShrink[c],
#     logDf[cols].mean(axis=1),
#     title=f'shrink_comp_result_{c}'
# )

    """
    plotName = f'Volcano plot - {title}'
    logger.info(f'Plotting {plotName}')
    plt.close(plotName)
    colFc = 'log2FC'
    colPv = 'adj.pvalue'
    if 'ImputationPercentage' in compDf.columns:
        prog = 'MSstats'
    elif 'baseMean' in compDf.columns:
        prog = 'DESeq2'
    else:
        raise

    directory = f'Plots/Volcano/'
    if not os.path.isdir(directory):
        os.makedirs(directory)
    name = f'volcano_{prog}_{title}'
    filePath = os.path.join(directory, name)
    timeStr = datetime.now().strftime("%Y%m%d%H%M%S")
    filePath += (timeStr if filePath[-1] in ['_', '/'] or os.path.isdir(filePath) else f'_{timeStr}')

    # clean up data
    compDf = compDf[~compDf[colFc].isna()]
    compDf = compDf[~compDf[colPv].isna()]
    commonIds = compDf.index.intersection(quantSeries.index)
    assert len(commonIds) > 0
    compDf = compDf.loc[commonIds, :]
    quantSeries = quantSeries[commonIds]

    # Get data to plot
    log2fc = compDf[colFc]
    pval = compDf[colPv]
    procPval = -np.log10(pval)
    logpThresh = -np.log10(pThresh)

    vDf = pd.concat([log2fc, procPval], axis=1, names=[colFc, f"'-log10({colPv})"])

    # Set positive and negative X range equal
    ymax = procPval[np.isfinite(procPval)].max() * 1.1
    yinf = procPval[~np.isfinite(procPval)].index
    ymin = 0 - ymax/1.1*0.05
    xmax = np.abs(log2fc[np.isfinite(log2fc)]).max() * 1.1
    # Move inf points to the edge of the plot
    log2fc = log2fc.replace(np.inf, xmax)
    log2fc = log2fc.replace(-np.inf, -xmax)
    procPval = procPval.replace(np.inf, ymax)
    assert all(log2fc.index == procPval.index)

    # Process quantification data
    normalisedQ = MinMaxScaler().fit_transform(quantSeries.to_numpy().reshape(-1, 1))
    quantSeries = pd.Series(normalisedQ.flatten(), index=quantSeries.index)
    # Calculate point sizes

    def calculateSize(x):
        return x * 50 + 1
    pointSizes = calculateSize(quantSeries)

    # Colours
    baseColor = 'C0'
    sigColor = 'C1'
    colours = pd.Series([baseColor]*log2fc.shape[0], index=log2fc.index)
    colours.loc[(~log2fc.between(-lfcThresh, lfcThresh, inclusive='neither')) & (procPval >= logpThresh)] = sigColor

    fig, ax = plt.subplots(1, 1, figsize=figsize, num=plotName)
    sc = ax.scatter(log2fc, procPval,
                    s=pointSizes, c=colours)
    ax.set_xlim((-xmax, xmax))
    ax.set_ylim((ymin, ymax))
#     ax.set_xlim((-xmax*1.01, xmax*1.01))
#     ax.set_ylim((ymin*1.01, ymax*1.01))
    ax.axhline(logpThresh, c='k', linestyle='--', linewidth=0.3)
    ax.axvline(-lfcThresh, c='k', linestyle='--', linewidth=0.3)
    ax.axvline(lfcThresh, c='k', linestyle='--', linewidth=0.3)
    ax.set_xlabel(r'$log_2$(fold change)')
    ax.set_ylabel(r'$-log_{10}$(adjusted P-value)')

    fixedXlim = True
    fixedYlim = True
    # hovering data and functions
    labels = log2fc.index
    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()
    if not fixedYlim:
        ymin, ymax = _extendRange((ymin, ymax), 1.1)
    if not fixedXlim:
        xmin, xmax = _extendRange((xmin, xmax), 1.1)
    lnv = ax.plot([0, 0], [ymin, ymax], color='black', linewidth=0.3)[0]
    lnh = ax.plot([xmin, xmax], [0, 0], color='black', linewidth=0.3)[0]
    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))
    lnv.set_linestyle('None')
    lnh.set_linestyle('None')
    annot = ax.annotate('', xy=(0, 0), xytext=(5, 5), textcoords="offset points")
    annot.set_visible(False)

    def hover(event):
        if event.inaxes == ax:
            lnv.set_data([event.xdata, event.xdata], [ymin, ymax])
            lnh.set_data([xmin, xmax], [event.ydata, event.ydata])
            lnv.set_linestyle('--')
            lnh.set_linestyle('--')
            lnv.set_visible(True)
            lnh.set_visible(True)
            # Test if on data points
            cont, ind = sc.contains(event)
            if cont:
                annot.xy = (event.xdata, event.ydata)
                annot.set_text('\n'.join([labels[n] for n in ind['ind']]))
                annot.set_visible(True)
            else:
                annot.set_visible(False)
        else:
            lnv.set_visible(False)
            lnh.set_visible(False)
    fig.canvas.mpl_connect("motion_notify_event", hover)

    if square:
        square_subplots(fig, ax)

    ax.set_title(name)

    fig.savefig(f'{filePath}.svg')
    vFilledDf = pd.concat([log2fc, procPval], axis=1)
    vFilledDf.columns = [c+'_filled' for c in vFilledDf.columns]
    vFilledDf.to_excel(f'{filePath}_filled.xlsx')
    vDf.to_excel(f'{filePath}.xlsx')



# Query subset


def query(meanDf, barDf, ids, conditions, figsize=(6, 4), title='', ylims=None, xs=None,
          plotType='bar', queryConditionGroupNames=None, xlabels=[], xlabelRotation=0):
    """
    experiments is a dict generated by loadMeta()
    conditions can be of maximal 2 deminsions
        eg. ['Dgbn_20', 'Dgbn_24', 'Dgbn_26', 'Dgbn_45', 'QC', 'WT_20', 'WT_24', 'WT_26', 'WT_45']
        or  [['Dgbn_20', 'Dgbn_24', 'Dgbn_26', 'Dgbn_45'],
             ['WT_20',   'WT_24',   'WT_26',   'WT_45'  ]]
    plotType in ['bar', 'line fill', 'line bar'], to be added: 'line 2D', "bar 2D"
    """
    plotName = f'query - {title} - {ids}'
    logger.info(f'Query with name: {plotName}')
    plt.close(plotName)
    # Normalise meanDf,
    Scaler = MinMaxScaler((0, 100))  # linear scaler
    Scaler.fit(meanDf)
    meanDf = pd.DataFrame(Scaler.transform(meanDf), index=meanDf.index, columns=meanDf.columns)
    barDf = pd.DataFrame(Scaler.transform(barDf), index=barDf.index, columns=barDf.columns)

    # Deal if given id is part of the id
    if isinstance(ids, str):
        ids = [ids]
    realIds = []
    for i in ids:
        corrId = meanDf.index[meanDf.index.str.contains(i)]
        if len(corrId) > 1:
            logger.info(f'Found multiple items for {i}: {list(corrId)}')
        elif len(corrId) == 0:
            logger.info(f'Item {i} not found in the target dataframe')
        realIds.extend(list(corrId))
    meanDf = meanDf.loc[realIds, :]
    barDf = barDf.loc[realIds, :]
#     print(meanDf)
#     print(barDf)

    conditions = np.array(conditions)
    groups = []
    cols = []
    assert conditions.ndim in [1, 2]
    if conditions.ndim == 2:
        assert len(realIds) == 1, f'Please chose only one to plot: {realIds}'
    else:
        conditions = conditions.reshape(1, -1)

    # deal with lables (names) input for plot
    if isinstance(queryConditionGroupNames, type(None)):
        # generate numbers as labels
        # if only one dimension, the names will be changed to column names later
        queryConditionGroupNames = [list(range(x)) for x in conditions.shape]
    qshape = tuple(len(x) for x in queryConditionGroupNames)
    assert qshape == conditions.shape, f'{qshape}, {conditions.shape}'

    # gather data
    for gname, subsetConds in zip(queryConditionGroupNames[0], conditions):
        subMeanDf = meanDf.loc[:, subsetConds]
        subBarDf = barDf.loc[:, subsetConds]
        groups.append([subMeanDf, subBarDf, gname])

    # plotting
    fig, ax = plt.subplots(1, 1, figsize=figsize, num=plotName)
    assert len(realIds) == 1 or len(groups) == 1
    if len(realIds) == 1:  # many groups, one id
        # replace column names:
        for subMeanDf, subBarDf, gname in groups:
            subMeanDf.columns = queryConditionGroupNames[1]
            subBarDf.columns = queryConditionGroupNames[1]
        meanDf = pd.concat([g[0] for g in groups], axis=0)
        meanDf.index = [g[2] for g in groups]
        barDf = pd.concat([g[1] for g in groups], axis=0)
        barDf.index = [g[2] for g in groups]
        ax.set_xlabel('Experiments')
        ax.set_ylabel(realIds[0])
    else:  # many ids, one group
        meanDf = groups[0][0]
        barDf = groups[0][1]
        ax.set_xlabel('Conditions/Strains')
    if isinstance(xs, type(None)) or plotType == 'bar':
        xs = np.arange(meanDf.shape[1])
    else:
        assert len(xs) == meanDf.shape[1]

    if plotType == 'bar':
        # make sure there is always a factor (ids or groups) that is 1 dimention
        width = 0.35
        singleBar = 2*width/meanDf.shape[0]
        for i, ind in enumerate(meanDf.index):
            #             print(meanDf)
            ax.bar(xs-width+i*singleBar, meanDf.loc[ind, :], label=ind,
                   yerr=barDf.loc[ind, :], width=singleBar, align='edge')
    elif plotType.startswith('line'):
        for i, ind in enumerate(meanDf.index):
            y = meanDf.loc[ind, :]
            b = barDf.loc[ind, :]
            y1 = y - b
            y2 = y + b
            if 'fill' in plotType:
                ax.plot(xs, y, label=ind)
                ax.fill_between(xs, y1, y2, alpha=0.3)
            elif 'bar' in plotType:
                ax.errorbar(xs, y, yerr=b, label=ind)
    else:
        raise ValueError(f"plotType should be one of ['bar', 'line fill', 'line bar'] not {plotType}")

    ax.set_xticks(xs)
    ax.set_xticklabels(meanDf.columns)
    if not isinstance(ylims, type(None)):
        ax.set_ylims(ylims)
    ax.legend()
    figtitle = f'Protein group{"s" if len(realIds) > 1 else ""} expression profile'
    figtitle += (f'\n{title}' if title != "" else "")
    ax.set_title(figtitle)
    if len(xlabels) != 0:
        if len(xlabels) == meanDf.shape[1]:
            ax.set_xticklabels(xlabels)
        else:
            logger.debug(f'xlabels invalid, should be of length {meanDf.shape[1]}, you passed {len(xlabels)}')

    for l in ax.get_xticklabels():
        l.set_rotation(xlabelRotation)
    plt.tight_layout()
    plt.show()

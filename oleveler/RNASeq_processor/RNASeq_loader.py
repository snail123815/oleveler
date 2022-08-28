import os

import pandas as pd
import numpy as np

from oleveler import calHash, logger
from .biology_information import _parseGff, _parseTableForGeneLength



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


def gatherCountTable(countsPath, toRemove=[]):
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
            if fn in toRemove:
                continue
            fp = os.path.join(countsPath, f)
            c = pd.read_csv(fp, sep='\t', comment='#',
                            header=0, index_col=0).iloc[:, 5]
            ct[fn] = c
    return ct


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
    ha = calHash(countTableDf, infoFiles,
                 tagsForGeneName,
                 lengthColParsingKeys, typeCol,
                 removerRNA, removeIDsubstrings, removeIDcontains)
    os.makedirs('dataTables', exist_ok=True)
    tpmTablePath = f'dataTables/TPM_Table_{ha}.tsv'
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
        # Read again from file ensuring the hash stays the same if reload
        tpmDf = pd.read_csv(tpmTablePath, sep='\t', index_col=0)
        logger.info(f'TPM data write to table {tpmTablePath}.')
    return tpmDf

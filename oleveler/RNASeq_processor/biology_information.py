import os
import pandas as pd
from BCBio import GFF
from oleveler import logger

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
                if 'gene_biotype' in feat.qualifiers:
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
    else:
        sep = ' '
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

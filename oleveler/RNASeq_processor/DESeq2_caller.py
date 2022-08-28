from tempfile import NamedTemporaryFile
import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError
import os
import sys
from collections import OrderedDict
from rpy2 import robjects
from rpy2.rinterface_lib.embedded import RRuntimeError
import rpy2.robjects.packages as rpackages
from oleveler import logger, calHash, loadMeta, genComparisonResults
from oleveler.main import _checkExistingCompResult
from threading import Thread
from multiprocessing import Process




# DESeq2 VST transformation
def genInfoDfDESeq2(dataDf, metaDf):
    '''
    dataDf.column should have same elements as metaDf.Experiment
    '''
    infoDf = metaDf.Condition.to_frame()
    infoDf.index = metaDf.Experiment
    infoDf = infoDf.reindex(index=dataDf.columns)
    tempInfoFile = NamedTemporaryFile(delete=False)
    infoDf.index = [c.replace('-','.') for c in infoDf.index]
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
    tempCleanDataFile = NamedTemporaryFile(delete=False)
    feedToRDf.columns = [c.replace('-','.') for c in feedToRDf.columns]
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
    If you want to do difference analysis, pass deOnly=True
    """
    pathTransformed = os.path.join('dataTables', 'transformed')
    os.makedirs(pathTransformed, exist_ok=True)
    # Remove columns based on metaDf
    newCol = []
    for e in metaDf.Experiment:
        if e in dataDf.columns:
            newCol.append(e)
        else:
            logger.info(f'{e} not in data table.')
    dataDf = dataDf.loc[:, newCol]

    # Calculate hash
    ha = calHash(dataDf, metaDf, ref, deOnly, designCol)

    logger.info(f'Current deseq2Process parameter hash = {ha}')

    pathVst = os.path.join(pathTransformed, f'vst_{ha}.tsv')
    pathVst = pathVst.replace('\\', '\\\\')

    if os.path.isfile(pathVst) and not deOnly:
        logger.info(f'Read VST transformed data from {pathVst}.')
        vstDf = pd.read_csv(pathVst, sep='\t', header=0, index_col=0)
        return vstDf

    logger.info('####### Process data using DESeq2 #######')
    tempDataFile = writeDataForDESeq2(dataDf)
    tempInfoFile, features = genInfoDfDESeq2(dataDf, metaDf)
    design = features[designCol]
    logger.info('Fitting data using DESeq2...')
    idxName = dataDf.index.name
    tif = tempInfoFile.name.replace('\\', '\\\\')
    tdf = tempDataFile.name.replace('\\', '\\\\')
    # Clean temp file (or Windows will raise "cannot find unused tempfile name"
    # after several loops)
    robjects.r(
        """
        rm(list = ls())
        sapply(file.path(tempdir(), list.files(tempdir())), unlink)
        gc()
        """
    )
    rpackages.importr("DESeq2")
    # Prepare DESeq2 input
    robjects.r(
        f"""
        coldata = read.csv('{tif}', sep='\t', row.names = 1, header = TRUE)
        cts = as.matrix(read.csv('{tdf}',sep='\t', row.names = '{idxName}'))
        """+"""
        if (any(rownames(coldata) != colnames(cts))){
            print("Column names not equal. Info file:")
            print(rownames(coldata))
            print("Data file:")
            print(colnames(cts))
        }
        """
    )

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

    try:
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
    except RRuntimeError as e:
        logger.info(f'Reading data with DESeq2 result error. The current environment:')
        logger.info('column data')
        logger.info(robjects.r('coldata'))
        logger.info('data table')
        logger.info(robjects.r('head(cts)'))
        raise(e)

    tempInfoFile.close()
    tempDataFile.close()
    os.remove(tempInfoFile.name)
    os.remove(tempDataFile.name)
    logger.info('END DESeq2 data init with current design:')
    if deOnly:
        return design
    else:
        logger.info('Performing VST')

        # Calculate nsub for vst
        robjects.r(
            """
            nsub <- 1000
            baseMean <- rowMeans(counts(dds, normalized=TRUE))
            max_nsub <- sum(baseMean > 5) 
            if (max_nsub < 1000) {
                nsub <- max_nsub
                print("Less than 1000 rows with mean normalized count > 5.")
                print(nsub)
            }
            """
        )

        robjects.r(
            f"""
            vsd = vst(dds, nsub=nsub, blind=FALSE)
            write.table(assay(vsd), file='{pathVst}', sep='\t', col.names=NA)
            rm(vsd)
            gc()
            """
        )
        logger.info(pathVst)
        logger.info('####### END Processing data using DESeq2 #######')
        vstDf = pd.read_csv(pathVst, sep='\t', header=0, index_col=0)
        # sort columns for the result.
        vstDf = vstDf.loc[:, sorted(vstDf.columns)]
        vstDf.to_csv(pathVst, sep='\t')
        vstDf = pd.read_csv(pathVst, sep='\t', header=0, index_col=0)
        return vstDf

def _deseq2Comp(a, b, name, extra, subResFileName):
    # First create environment for use in seperate threads or processes
    rstr = f"""
        #print(resultsNames(dds))
        res <- {a}(dds, {b}="{name}", parallel=TRUE{extra})
        
        write.table(as.data.frame(res), col.names=NA, sep='\t',
                    file="{subResFileName}")
        #print("{subResFileName}")
        """
    # Do not directly use f string in r() function here, for maximal compatibility with
    # Thread() in Windows
    robjects.r(rstr)


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
        logger.info(f'Calculate comparisons {exp} vs. {ctr}.')
        if ctr == previousCtr:
            design = previousDesign
        else:
            design = deseq2Process(countTable, metaDf, ref=ctr, deOnly=True)
            previousDesign = design
            previousCtr = ctr
        name = f'{design}_{exp}_vs_{ctr}'
        subResFile = NamedTemporaryFile(delete=False)
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
        #fn = subResFile.name
        #print('temp file name ', fn)
        subResFile.close()
        #print('temp file name after close ', subResFile.name)
        
        while os.path.getsize(subResFile.name) == 0:
            # print(os.path.getsize(subResFile.name))
            if i > 0:
                logger.info(f'Failed to calculate comparison within {int(t/60)} min, retry.')
            srf = subResFile.name.replace('\\', '\\\\')

            # Add time out to function. Trust me with Thread or Process!!!
            # _deseq2Comp() is an function calling R code by rpy2, needs to communicate with the
            # only r kernel managed by rpy2. Add time out is just to retry when r kernel do not
            # respond.
            # In *nix system with Thread(), the died kernel stays no respond, so Process() needed.
            # In Win, Process() will lose the contact with rpy2 kernel, only Thread() is safe and
            # do not have the problem of no responding r kernel (or do not have the problem like
            # above).
            # Maybe experiment with spawn and fork?
            if sys.platform.startswith("win") or sys.platform == 'darwin':
                th = Thread(target=_deseq2Comp, args=(a, b, name, extra, srf), daemon=True)
                th.start()
                th.join(t)
            else:
                p = Process(target=_deseq2Comp, args=(a, b, name, extra, srf), daemon=True)
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
        os.remove(subResFile.name)
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


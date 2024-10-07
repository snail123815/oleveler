import os
import pandas as pd
import rpy2.robjects.packages as rpackages
from rpy2.rinterface_lib.embedded import RRuntimeError
import rpy2.robjects as robjects
from oleveler import calHash
from oleveler.safe_annotations import safeCol, safeAnnotations, safeMQdata, logger
from oleveler.comparison_processor import _checkExistingCompResult, genComparisonResults
from oleveler.metadata_processor import loadMeta
from .MaxQuant_data_loader import loadMQLfqData

# MSstats proposition


def clearlogfiles(msstatsLogPath='dataTables/transformed/MSstats_log/'):
    os.makedirs(msstatsLogPath, exist_ok=True)
    # move log file to log folder
    for f in os.listdir():
        if f.startswith('MSstats_') and f.endswith('.log'):
            os.rename(f, os.path.join(msstatsLogPath, f))


def prepareMSstats(mqDataPath, annotationPath, toRemove=[]):
    """
    This will be reused in calculating differences if has not done in current session.
    """

    logger.info('####### MSstats import and proposition #######')

    logger.info('Import MSstats')
    rpackages.importr("MSstats")
    pgPath = os.path.join(mqDataPath, "proteinGroups.txt")
    evPath = os.path.join(mqDataPath, 'evidence.txt')
    logger.info('Read MaxQuant data')
    annotationPath = safeAnnotations(annotationPath, toRemove=toRemove)
    experiments = pd.read_csv(annotationPath, header=0, usecols=['Experiment']).Experiment.unique()
    annotationPath = annotationPath.replace('\\', '\\\\')
    pgPath = pgPath.replace('\\', '\\\\')
    evPath = evPath.replace('\\', '\\\\')
    evExperiments = safeCol(pd.read_csv(evPath, sep='\t', usecols=['Experiment']).Experiment.unique())
    if not all(exp in experiments for exp in evExperiments):
        toRemove = [exp for exp in evExperiments if exp not in experiments]
    pgPath, evPath = safeMQdata(pgPath, evPath, toRemove=toRemove)
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


def processMSstats(mqDataPath, annotationPath, toRemove=[]):
    # calculate Hash
    pgPath = os.path.join(mqDataPath, "proteinGroups.txt")
    evPath = os.path.join(mqDataPath, 'evidence.txt')
    ha = calHash(pgPath, evPath, annotationPath, toRemove)

    annotationPath = safeAnnotations(annotationPath, toRemove)

    qcplotPath = f'dataTables/transformed/msstats_proposed_{ha}_QCPlot.pdf'
    proposedDataPath = f'dataTables/transformed/msstats_proposed_{ha}.tsv'
    logDataPath = f'dataTables/transformed/msstats_proposed_{ha}_logTable.tsv'

    # Calculate quantitation table
    if os.path.isfile(proposedDataPath):
        logger.info('Read proposed data from table')
        logger.info(proposedDataPath)
        quantTable = pd.read_csv(proposedDataPath, sep='\t')
    else:
        prepareMSstats(mqDataPath, annotationPath)
        logger.info('Write proposed data to table')
        logger.info(proposedDataPath)
        proposedDataPath = proposedDataPath.replace('\\', '\\\\')
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
        logger.info(logDataPath)
        logDf = pd.read_csv(logDataPath, sep='\t', index_col=0)
    else:
        logger.info('Transform MSstats proposed data into data table')
        logger.info(logDataPath)
        pgIds = pd.unique(quantTable.Protein)  # protein groups
        pgIds.sort()
        runIds = pd.unique(quantTable.originalRUN)
        runIds.sort()
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


def msstatsComp(comparisons, mqDataPath, annotationPath, compResultFile, toRemove=[]):
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
        prepareMSstats(mqDataPath, annotationPath, toRemove=toRemove)
        robjects.r(
            """
            colnames(comparisons) <- levels(maxquant.proposed$ProteinLevelData$GROUP)
            """
        )
    logger.info(f'Calculating comparisons...')
    compResultFile = compResultFile.replace('\\', '\\\\')
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



def makeCompMatrixMsstats(compExcel, mqDataPath, annotationPath, toRemove=[]):
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

    lfqDf, _ = loadMQLfqData(mqDataPath, toRemove=toRemove)  # only used for check if result exists, to avoid extensive R calls
    allCompResults, comparisons, compResultFile = \
        _checkExistingCompResult(compExcel, lfqDf, compResultFileBase)
    if not isinstance(allCompResults, type(None)):
        return allCompResults, comparisons

    logger.info(f'Parse {compExcel}, generate comparison matrix for MSstats.')
    compResultFile = msstatsComp(comparisons, mqDataPath, annotationPath, compResultFile, toRemove=toRemove)
    allCompResults = genComparisonResults(compResultFile, comparisons)
    logger.info(f'Done calculation, dump data in {compResultFile}\n')
    clearlogfiles()
    return allCompResults, comparisons
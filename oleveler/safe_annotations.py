import os
import re
from tempfile import NamedTemporaryFile
import pandas as pd
from pathlib import Path
from .project_logger import logger


def safeCol(cols):
    illegal = re.compile(r"[ \-:,();{}+*'\"[\]\/\t\n]")
    return [illegal.sub("_", col) for col in cols]


def safeExperimentNameMQ(MQDataPath): # Not used, but have a test case.
    '''
    Check if the 'experiments' column contains `-`, if so, change to `.`.
    This change should ensure success run of DESeq2'''
    files = os.listdir(MQDataPath)
    assert all(f in files for f in ['evidence.txt', 'proteinGroups.txt'])

    # evidence.txt
    evFile = os.path.join(MQDataPath, 'evidence.txt')
    evDf = pd.read_csv(evFile, sep='\t', header=0, index_col=None)
    assert all(c in evDf.columns for c in ['Raw file', 'Experiment']), \
        f'Does not find "Raw file", "Experiment" column in {evFile}.\n{evDf.head()}'
    experiments = evDf['Experiment'].unique()
    illegal = re.compile('-+')
    hasIllegal = False
    for exp in experiments:
        if illegal.search(exp):
            hasIllegal = True
            break
    if hasIllegal:
        logger.warning(f'Find illegal pattern "{illegal.pattern}" in Experiment setup, ' + \
            'will replace it with "_"')
        evDf['Experiment'] = evDf['Experiment'].map(lambda x: illegal.sub('_', x))
        os.rename(evFile, evFile+'._bk')
        evDf.to_csv(evFile, sep='\t', index=False)
        pgFile = os.path.join(MQDataPath, 'proteinGroups.txt')
        pgDf = pd.read_csv(pgFile, sep='\t', header=0, index_col=0)
        newColumns = []
        for col in pgDf.columns:
            for exp in experiments:
                if exp in col:
                    cExp = illegal.sub('_', exp)
                    col = col.replace(exp, cExp)
                    break
            newColumns.append(col)
        pgDf.columns = newColumns
        os.rename(pgFile, pgFile+'._bk')
        pgDf.to_csv(pgFile, sep='\t')
    return


def safeAnnotations(annotationPath: Path, toRemove=[]):
    """
    annotationPath: str, path to the annotation file
    toRemove: list, list of experiments to remove from the annotation file

    annotation file has to be a csv file, for the use in R DESeq2
    The column names are: 'Raw.file', 'Condition', 'BioReplicate', 'Experiment'
    'Raw.file' and 'Experiment' will be the same in the file if
    `lcmsms_randomasiation.tsv` is used to change data table header.

    Cannot change these columns because R package DESeq2 requires them.

    ```R
    annot <- read.csv("{annotationPath}", header=TRUE)
    ```

    TODO use lcmsms_randomasiation.tsv to generate annotation file.

    """
    annoDf = pd.read_csv(
        annotationPath,
        header=0,
        sep=",",
        usecols=["Raw.file", "Condition", "BioReplicate", "Experiment"],
        index_col="Raw.file",
    )
    # Assert index column only contains unique values
    duplicated_index = annoDf.index[annoDf.index.duplicated()].tolist()
    if len(duplicated_index) > 0:
        logger.error(
            f"Index column contains duplicated values {duplicated_index}"
        )
        raise ValueError(
            f"Index column contains duplicated values {duplicated_index}"
        )
    needs_change = False
    if len(toRemove) != 0:
        needs_change = True
        for r in toRemove:
            assert (
                r in annoDf["Experiment"].values
            ), f"{r} not in {annoDf.Experiment}"
            annoDf = annoDf[annoDf.Experiment != r]
    safeConds = safeCol(annoDf["Condition"])
    safeExpes = safeCol(annoDf["Experiment"])
    if not all(c in annoDf["Condition"] for c in safeConds) or not all(
        e in annoDf["Experiment"] for e in safeExpes
    ):
        needs_change = True
    if needs_change:
        annoDf["Condition"] = safeConds
        annoDf["Experiment"] = safeExpes
        with NamedTemporaryFile(delete=False) as anSafe:
            annoDf.to_csv(anSafe.name)
            annotationPath = anSafe.name

    return annotationPath


def safeMQdata(pgPath, evPath, toRemove=[]):
    experiments = pd.read_csv(evPath, sep='\t', usecols=['Experiment'])['Experiment'].unique()
    safeExps = safeCol(experiments)
    if not all(e in experiments for e in safeExps) or len(toRemove) != 0:
        global pgSafe
        global evSafe
        pgSafe = NamedTemporaryFile()
        evSafe = NamedTemporaryFile()

    if not all(e in experiments for e in safeExps):
        with open(evPath, 'r') as oev:
            with open(evSafe.name, 'w') as nev:
                headers = oev.readline()
                expIdx = headers.split('\t').index('Experiment')
                nev.write(headers)
                for l in oev:
                    row = l.split('\t')
                    if row[expIdx] in toRemove:
                        continue
                    try:
                        row[expIdx] = safeExps[experiments.tolist().index(row[expIdx])]
                        if row[expIdx] in toRemove:
                            continue
                    except ValueError as e:
                        raise e
                    nev.write('\t'.join(row))

        with open(pgPath, 'r') as opg:
            with open(pgSafe.name, 'w') as npg:
                headers = opg.readline().split('\t')
                nheaders = []
                toRemoveCols = []
                for i, h in enumerate(headers):
                    needRemoval = False
                    ts = h.split(' ') # The experiment cannot contain space
                    if ts[-1] in toRemove:
                        toRemoveCols.append(i)
                        needRemoval = True
                    try:
                        ts[-1] = safeExps[experiments.tolist().index(ts[-1])]
                        if not needRemoval and ts[-1] in toRemove:
                            toRemoveCols.append(i)
                            needRemoval = True
                    except ValueError:
                        pass
                    if not needRemoval:
                        nheaders.append(' '.join(ts))
                npg.write('\t'.join(nheaders))
                if len(toRemoveCols) == 0:
                    npg.writelines(opg.readlines())
                else:
                    for l in opg:
                        eles = l.split('\t')
                        npg.write(
                            '\t'.join(eles[i] for i in range(len(eles)) if i not in toRemoveCols)
                        )

        evPath = evSafe.name
        pgPath = pgSafe.name

    elif len(toRemove) != 0:
        with open(evPath, 'r') as oev:
            with open(evSafe.name, 'w') as nev:
                headers = oev.readline()
                expIdx = headers.split('\t').index('Experiment')
                nev.write(headers)
                for l in oev:
                    row = l.split('\t')
                    if row[expIdx] in toRemove:
                        continue
                    nev.write(l)

        with open(pgPath, 'r') as opg:
            with open(pgSafe.name, 'w') as npg:
                headers = opg.readline().split('\t')
                nheaders = []
                toRemoveCols = []
                for i, h in enumerate(headers):
                    ts = h.split(' ') # Experiment name cannot contain space
                    if ts[-1] in toRemove:
                        toRemoveCols.append(i)
                        continue
                    nheaders.append(' '.join(ts))
                npg.write('\t'.join(nheaders))
                if len(toRemoveCols) == 0:
                    npg.writelines(opg.readlines())
                else:
                    for l in opg:
                        eles = l.strip().split('\t')
                        npg.write(
                            '\t'.join(eles[i] for i in range(len(eles)) if i not in toRemoveCols)
                        )

        evPath = evSafe.name
        pgPath = pgSafe.name

    return pgPath, evPath

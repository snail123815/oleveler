import os
import re
from tempfile import NamedTemporaryFile
import pandas as pd
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


def safeAnnotations(annotationPath, toRemove=[]):
    conditions = pd.read_csv(annotationPath, usecols=['Condition'])['Condition'].unique()
    safeConds = safeCol(conditions)
    if not all(c in conditions for c in safeConds):
        global anSafe_1
        anSafe_1 = NamedTemporaryFile()
        with open(annotationPath, 'r') as oan:
            with open(anSafe_1.name, 'w') as nan:
                nan.write(oan.readline())
                for l in oan:
                    row = l.split(',')
                    try:
                        row[1] = safeConds[conditions.tolist().index(row[1])]
                    except ValueError as e:
                        raise e
                    except IndexError as e:
                        raise e
                    nan.write(','.join(row))
        annotationPath = anSafe_1.name
    experiments = pd.read_csv(annotationPath, usecols=['Experiment'])['Experiment'].unique()
    safeExps = safeCol(experiments)
    if not all(e in experiments for e in safeExps):
        global anSafe_2
        anSafe_2 = NamedTemporaryFile()
        with open(annotationPath, 'r') as oan:
            with open(anSafe_2.name, 'w') as nan:
                nan.write(oan.readline())
                for l in oan:
                    row = l.split(',')
                    try:
                        row[3] = safeExps[experiments.tolist().index(row[3].strip())]
                    except ValueError as e:
                        raise e
                    except IndexError as e:
                        raise e
                    if not row[-1].endswith('\n'):
                        row[-1] += "\n"
                    nan.write(','.join(row))
        annotationPath = anSafe_2.name
        anSafe_1.close()
    if len(toRemove) != 0:
        annDf = pd.read_csv(annotationPath, index_col=0, header=0)
        for r in toRemove:
            assert r in annDf.Experiment.values, f"{r} not in {annDf.Experiment}"
            annDf = annDf[annDf.Experiment != r]
        global anSafe_3
        anSafe_3 = NamedTemporaryFile()
        annDf.to_csv(anSafe_3.name)
        annotationPath = anSafe_3.name
        anSafe_2.close()

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

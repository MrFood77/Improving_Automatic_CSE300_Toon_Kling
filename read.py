import pandas as pd


# Transform files into project id's.
def getProj(name: str):
    part1 = name[32:-1]
    part2 = part1[0:part1.find('/')]
    return part2


def getData():
    # Read the data from a file
    classData = pd.read_csv('./data/class.csv')
    columns = list(classData.columns)

    # Remove string values
    columns.remove('type')

    all_data = classData.loc[:, columns]
    # Turn the file names into the proj id, which allows joins with the results.
    all_data.loc[:, 'file'] = all_data.loc[:, 'file'].map(getProj)
    # make joining easier.
    all_data = all_data.rename(columns={'file': 'project.id', 'class': 'TARGET_CLASS'})

    return all_data


def getBranchCoverage():
    # The columns that we are interested in
    relevantColumns = ['BranchCoverage', 'criterion', 'project.id', 'TARGET_CLASS', 'round']

    data300 = pd.read_csv('./data/results-300.csv')[relevantColumns]
    data300['dataset'] = "300"
    data180 = pd.read_csv('./data/results-180.csv')[relevantColumns]
    data180['dataset'] = "180"
    data60 = pd.read_csv('./data/results-60.csv')[relevantColumns]
    data60['dataset'] = "60"
    data_new = pd.concat([data300, data180, data60]).sort_values(['project.id', 'TARGET_CLASS', 'criterion', 'round', 'dataset'])
    data_new = data_new.rename(columns={'BranchCoverage': 'Score'})
    # This is so that the data looks the same way as the mutation score:
    data_new.loc[data_new['criterion'] == "LINE;BRANCH;EXCEPTION;WEAKMUTATION;OUTPUT;METHOD;METHODNOEXCEPTION;CBRANCH", 'criterion'] = "default"
    return data_new


def getMutationScore():
    df = pd.read_csv('./data/mutation_scores.csv')
    df = df.rename(columns={'project': 'project.id', 'class': 'TARGET_CLASS', 'mutation_score_percent': 'Score'})
    df['criterion'] = df['configuration'].map(configToCriterion)
    df['dataset'] = df['configuration'].map(configToDataSet)
    rel_cols = list(df.columns)
    rel_cols.remove("killed_mutants")
    rel_cols.remove("total_mutants")
    df = df[rel_cols]
    return df


def configToCriterion(config):
    res = config[0:config.find('_')]
    if res == "exception":
        res = "BRANCH;EXCEPTION"
    if res == "branch":
        res = "BRANCH"
    if res == "input":
        res = "BRANCH;INPUT"
    if res == "output":
        res = "BRANCH;OUTPUT"
    if res == "weak":
        res = "BRANCH;WEAKMUTATION"
    return res


def configToDataSet(config):
    res = config[config.find('_')+1:]
    return res


def combine(a: pd.DataFrame, b: pd.DataFrame):
    return pd.merge(a, b, on=['project.id', 'TARGET_CLASS'])


def getCombined():
    return combine(getBranchCoverage(), getData())

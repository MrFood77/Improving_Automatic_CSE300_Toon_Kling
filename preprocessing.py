from pandas import DataFrame
import numpy as np
import filter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def roundRounds(df: DataFrame):
    res = df.groupby(['project.id', "TARGET_CLASS", "criterion", "dataset"]).median()
    res = res.reset_index()
    return res


def pca(df: DataFrame, n):
    pca = PCA(n_components=n)
    res = pca.fit_transform(df)
    return res


def getDiff(df: DataFrame):
    # diff = exception - branch!! positive -> exception is better
    diff = df.sort_values(['project.id', 'TARGET_CLASS', 'dataset'])
    diff['Score'] = diff['Score'].diff()
    crit = diff['criterion'].iloc[1]
    return filter.filterCriterion(diff, [crit]) \
        .rename(columns={"Score": "diff"}) \
        .loc[:, diff.columns != 'round']


def setIsBWE(df: DataFrame):
    conditions = [df['diff'] < 0, df['diff'] == 0, df['diff'] > 0]
    choices = ['better', 'equal', 'worse']
    df['Outperforming'] = np.select(conditions, choices)
    return df


def setIsBetter(df: DataFrame):
    df['Outperforming'] = df['diff'] < 0
    return df


def fillNan(df: DataFrame):
    df.loc[:, ['lcom*', 'tcc', 'lcc']] = df[['lcom*', 'tcc', 'lcc']].fillna(1)
    return df


def scale(df: DataFrame):
    rel_cols = list(df.columns)
    rel_cols.remove('project.id')
    rel_cols.remove('TARGET_CLASS')
    rel_cols.remove('diff')
    df[rel_cols] = StandardScaler().fit_transform(df[rel_cols])
    return df


def combineRows(df: DataFrame, baseCrit, compCrit):
    cols = list(df.columns)
    cols.remove('criterion')
    cols.remove('Score')
    cols.remove('round')
    cols.remove('dataset')
    tmp = df.pivot_table('Score', cols, 'criterion')
    tmp.reset_index(inplace=True)
    tmp['diff'] = tmp[baseCrit] - tmp[compCrit]
    return tmp

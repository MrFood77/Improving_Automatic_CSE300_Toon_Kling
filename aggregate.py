from pandas import DataFrame


def avgPerDataSet(df: DataFrame):
    df = df.groupby(['dataset', 'criterion']).mean()
    df = df.reset_index()
    return df

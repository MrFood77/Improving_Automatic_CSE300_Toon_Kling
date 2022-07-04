from pandas import DataFrame


def filterCriterion(data: DataFrame, criteria: [str]):
    return data[data['criterion'].isin(criteria)]


def filterCol(df: DataFrame, filterName: str):
    df = df.drop(filterName, axis=1)
    return df


def filterDataSet(df: DataFrame, dataset: str):
    return df[df['dataset'] == dataset]

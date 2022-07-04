from pandas import DataFrame
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, ADASYN


def simpleSplit(df: DataFrame, y_stat='Outperforming'):
    """
    Returns 4 dataframes, X_train, y_train, X_test and y_test according to an
    80% split.
    """
    X = df.loc[:, df.columns != y_stat]
    y = df[y_stat]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    return X_train, X_test, y_train, y_test


def overSample(X_train, X_test, y_train, y_test):
    over_sampler = RandomOverSampler()
    X_res, y_res = over_sampler.fit_resample(X_train, y_train)
    return X_res, X_test, y_res, y_test


def overSampleSMOTE(X_train, X_test, y_train, y_test):
    cols = list(set(X_train.columns) - {'TARGET_CLASS', 'project.id'})
    X_res, y_res = SMOTE().fit_resample(X_train[cols], y_train)
    return X_res, X_test, y_res, y_test


def overSampleADASYN(X_train, X_test, y_train, y_test):
    cols = list(set(X_train.columns) - {'TARGET_CLASS', 'project.id'})
    X_res, y_res = ADASYN(sampling_strategy='minority').fit_resample(X_train[cols], y_train)
    return X_res, X_test, y_res, y_test


def underSample(X_train, X_test, y_train, y_test):
    under_sampler = RandomUnderSampler()
    X_res, y_res = under_sampler.fit_resample(X_train, y_train)
    return X_res, X_test, y_res, y_test

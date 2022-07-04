from sklearn.model_selection import KFold
from sklearn import ensemble
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
import read
import filter
import preprocessing


def getLabeledData(scores, baseCrit, compCrit):
    data = read.getData()
    cols = list(set(data.columns) - {'project.id', 'TARGET_CLASS'})
    data['lcom*'].fillna(1, inplace=True)
    data['tcc'].fillna(-1, inplace=True)
    data['lcc'].fillna(-1, inplace=True)
    data = pd.concat([data[['project.id', 'TARGET_CLASS']], data[cols]], axis=1)
    combined = read.combine(scores, data)
    combinedFiltered = filter.filterCriterion(combined, [baseCrit, compCrit])
    medianed = preprocessing.roundRounds(combinedFiltered)
    df = preprocessing.combineRows(medianed, baseCrit, compCrit)
    labeled = preprocessing.setIsBWE(df)
    cool_cols = list(labeled.columns)
    cool_cols.remove('project.id')
    cool_cols.remove('diff')
    labeled = labeled[cool_cols]
    df = labeled.dropna()
    df = df.reset_index(drop=True)
    return df


def newNeuralMut(scores, baseCrit, compCrit):
    df = getLabeledData(scores, baseCrit, compCrit)
    split_cols = list(set(df.columns) - {'criterion', 'TARGET_CLASS', baseCrit, compCrit, 'Outperforming'})

    X = df.loc[:, df.columns != 'Outperforming']
    y = df['Outperforming']
    kf = KFold(n_splits=5, shuffle=True)
    dfs = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model = ensemble.RandomForestClassifier(max_depth=5, n_estimators=500, criterion='entropy')
        pipe = make_pipeline(StandardScaler(), model)
        pipe = pipe.fit(X_train[split_cols], y_train)

        pred_classes = pipe.predict(X_test[split_cols])

        # Transform the predicted values into True or falses
        res_df = X_test.join(y_test)
        res_df['pred'] = pred_classes

        dfs.append(res_df)

    res = pd.concat(dfs)
    res['Outperforming'] = res['Outperforming'].isin(["better", "equal"])
    res['pred'] = res['pred'].isin(["better", "equal"])
    return res

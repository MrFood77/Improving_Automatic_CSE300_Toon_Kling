from sklearn import metrics


def printAccuracy(y_pred, y_test):
    accuracy = accuracy = 1 - ((y_test != y_pred).sum() / int(y_test.shape[0]))
    print(f'accuracy: {accuracy}')


def printf1(y_pred, y_test):
    f1 = metrics.f1_score(y_test, y_pred, zero_division=0)
    print(f'f1: {f1}')


def printRecallSupport(y_pred, y_test):
    precision, recall, fbeta_score, support = metrics.precision_recall_fscore_support(y_pred, y_test)
    print(f'precision: {precision}')
    print(f'recall: {recall}')
    print(f'fbeta_score: {fbeta_score}')
    print(f'support: {support}')


def avgPerformance(data):
    print(f'Average performance of dataset with exception: '
          f'{data[data["criterion"] == "BRANCH;EXCEPTION"]["BranchCoverage"].mean()}')
    print(f'Average performance of dataset without excepts: '
          f'{data[data["criterion"] == "BRANCH"]["BranchCoverage"].mean()}')


def avgPerformanceWithMask(data, mask, baseCrit, compCrit):
    total = data[data['TARGET_CLASS'].isin(mask)][compCrit].sum() + data[~data['TARGET_CLASS'].isin(mask)][baseCrit].sum()
    res = total / data.shape[0]
    return res


def isOurChoiceBetterMut(df, baseCrit, compCrit):
    res1 = avgPerformanceWithMask(df, [], baseCrit, compCrit)
    res2 = avgPerformanceWithMask(df, df['TARGET_CLASS'], baseCrit, compCrit)
    res3 = avgPerformanceWithMask(df, df[df["pred"]]['TARGET_CLASS'], baseCrit, compCrit)
    res4 = avgPerformanceWithMask(df, df[df['Outperforming']]['TARGET_CLASS'], baseCrit, compCrit)

    diff = res3 - res2
    print(f'Difference in performance between compCrit and model: {diff}')

    return [res1, res2, res3, res4]

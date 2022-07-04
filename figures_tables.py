import pandas as pd
import read
import draw
import preprocessing
import aggregate
import filter
import model
import result


# A random sample of EvoSuite data, 𝑛 = 5
def table1():
    print("\n")
    relevantColumns = ['BranchCoverage', 'criterion', 'project.id', 'TARGET_CLASS', 'round']
    data = pd.read_csv('./data/results-300.csv')[relevantColumns].sample(5)
    print(data)


# or how many classes the different criterion outperform each other.
# Comparing BRANCH (B) to BRANCH;EXCEPTION (B;E) for both metrics that were collected.
def table2():
    print("\n")
    table2_helper(read.getBranchCoverage(), "Branch Coverage")
    table2_helper(read.getMutationScore(), "Mutation Score")


def table2_helper(bc, is_for):
    print("\n")
    print(f"--------------\nFor {is_for}:")
    bc = bc[bc['criterion'].isin(['BRANCH', 'BRANCH;EXCEPTION'])]
    bc = bc.groupby(['project.id', 'TARGET_CLASS', 'criterion', 'dataset']).median().reset_index()
    cols = list(set(bc.columns) - {'criterion', 'Score', 'round', 'dataset'})
    bc = bc.pivot_table('Score', cols, 'criterion').reset_index()

    print(f'Total amount of rows: {bc.shape[0]}')

    branchBetter = bc[bc['BRANCH'] > bc['BRANCH;EXCEPTION']].shape[0]
    print(f'Amount of times using branch is better: {branchBetter}')
    exceptBetter = bc[bc['BRANCH'] < bc['BRANCH;EXCEPTION']].shape[0]
    print(f'Amount of times using branch;except is better: {exceptBetter}')
    equal = bc[bc['BRANCH'] == bc['BRANCH;EXCEPTION']].shape[0]
    print(f'Amount of they are equal: {equal}')


# Most correlating metrics for Mutation Score
def table3():
    print("\n")
    table3_4_helper(read.getMutationScore())


# Most correlating metrics for Branch Coverage
def table4():
    print("\n")
    table3_4_helper(read.getBranchCoverage())


def table3_4_helper(data):
    df = model.getLabeledData(data, "BRANCH", "BRANCH;EXCEPTION")
    df['Outperforming'] = df['Outperforming'].isin(['better', 'equal'])
    data_cols = list(set(df.columns) - {'TARGET_CLASS', 'BRANCH', 'BRANCH;EXCEPTION'})
    corr = df[data_cols].corr()['Outperforming'].sort_values(key=abs)
    print(corr.tail(12))


# Average Branch coverage for each dataset and each criterion.
def fig1():
    print("\n")
    fig1_2_helper(read.getBranchCoverage())


# Average Mutation Score for both criteria.
def fig2():
    print("\n")
    fig1_2_helper(read.getMutationScore())


def fig1_2_helper(score):
    data = read.getData()
    combined = read.combine(score, data)
    combinedFiltered = filter.filterCriterion(combined, ['BRANCH', 'BRANCH;EXCEPTION'])
    medianed = preprocessing.roundRounds(combinedFiltered)
    avgd = aggregate.avgPerDataSet(medianed)

    draw.makeHistogram(avgd)


# Average Branch Coverage comparison for each possible criterion.
def fig3():
    print("\n")
    baseCrit = "BRANCH"
    compCrit = "BRANCH;EXCEPTION"
    res = model.newNeuralMut(read.getBranchCoverage(), baseCrit, compCrit)
    result.printAccuracy(res['pred'], res['Outperforming'])
    result.printf1(res['pred'], res['Outperforming'])
    result.printRecallSupport(res['pred'], res['Outperforming'])

    results = result.isOurChoiceBetterMut(res, baseCrit, compCrit)
    draw.comparePerformance(results, baseCrit, compCrit, offset=0.005)


# Average Mutation Score comparison for each possible criterion.
def fig4():
    print("\n")
    baseCrit = "BRANCH"
    compCrit = "BRANCH;EXCEPTION"
    res = model.newNeuralMut(read.getMutationScore(), baseCrit, compCrit)

    result.printAccuracy(res['pred'], res['Outperforming'])
    result.printf1(res['pred'], res['Outperforming'])
    result.printRecallSupport(res['pred'], res['Outperforming'])

    results = result.isOurChoiceBetterMut(res, baseCrit, compCrit)
    draw.comparePerformance(results, baseCrit, compCrit, offset=1)


table1()
table2()
table3()
table4()
fig1()
fig2()
fig3()
fig4()

exit()

# To ensure that the results are reproducable, the expriment was repeated 10 times.
# To make reading results easier, remove all print statementes except the statement
# that prints the difference.
print('For Branch Coverage:')
for i in range(10):
    fig3()

print('For Mutation Score:')
for i in range(10):
    fig4()

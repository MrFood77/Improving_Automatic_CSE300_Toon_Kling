import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame


def makeHistogram(data):
    plt.rcParams.update({'font.size': 22})

    labels = data['dataset'].unique()
    branch_means = data[data['criterion'] == "BRANCH"]['Score']
    branch_means = branch_means.round(3)
    excpt_means = data[data['criterion'] == "BRANCH;EXCEPTION"]['Score']
    excpt_means = excpt_means.round(3)

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, branch_means, width, label='BRANCH')
    rects2 = ax.bar(x + width/2, excpt_means, width, label='BRANCH;EXCEPTION')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Branch coverage')
    ax.set_title('Branch coverage by dataset and criterion')
    ax.set_xticks(x, labels)
    ax.legend(loc="lower right")

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    ax.set_ylim([0, max(branch_means)*1.5]) # just something to make all graphs more readable

    # fig.tight_layout()

    plt.show()


def makePlot(criteria, wilcoxon):
    x = np.arange(len(criteria))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rect = ax.bar(x, wilcoxon, width, label=criteria)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Wilcoxon scores')
    ax.set_title('Wilcoxon score for each criteria')
    ax.set_xticks(x, criteria)
    ax.set_xticklabels(criteria, rotation=70)

    ax.bar_label(rect, padding=3)

    plt.show()


def drawCorrMatrix(df: DataFrame):
    plt.matshow(df)
    plt.xlabel("Class metrics")
    plt.ylabel("Class metrics")
    plt.show()


def makeBoxPlot(data):
    print(f'data: {data}')
    labels = data['dataset'].unique()

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, [ax1, ax2, ax3] = plt.subplots(1, 3)
    comb_data = [data[data['criterion'] == "BRANCH"]['BranchCoverage'],
                 data[data['criterion'] == "BRANCH;EXCEPTION"]['BranchCoverage']]
    ax1.set_title('all data')
    ax1.boxplot(comb_data)

    plt.show()


def comparePerformance(data, baseCrit, compCrit, offset):
    plt.rcParams.update({'font.size': 18})
    plt.figure()
    names = [f'"{baseCrit}"', f'"{compCrit}"', 'Model predictions', 'Perfect predictions']

    plt.bar(names, data)
    plt.xlabel("Criteria")
    plt.ylabel("Average Mutation Score achieved")
    plt.title("Compare performance of different selection methods")
    plt.subplots_adjust(bottom=0.2)

    ax = plt.gca()
    ax.set_ylim([min(data) - offset, max(data) + offset])

    plt.xticks(rotation=10)
    plt.show()

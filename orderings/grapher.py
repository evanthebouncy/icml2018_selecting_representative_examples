import pickle
import matplotlib.pyplot as plt
import numpy as np

# font size setup
plt.rc('font', size=8)          # controls default text sizes
plt.rc('axes', titlesize=8)     # fontsize of the axes title
plt.rc('axes', labelsize=10)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=8)    # fontsize of the tick labels
plt.rc('ytick', labelsize=8)    # fontsize of the tick labels
plt.rc('legend', fontsize=8)    # legend fontsize
plt.rc('figure', titlesize=12)  # fontsize of the figure title


def graph_stuff(TEST_LOC='results.p'):
    data = pickle.load( open( TEST_LOC, "rb" ) )
    
    names = []
    num_examples = {}
    is_rep = {}
    p_rep = {}

    for i, trial in enumerate(data):
        for name, stats in trial.iteritems():
            if i == 0:
                names.append(name)
                num_examples[name] = [stats['num_examples']]
                is_rep[name] = [stats['is_rep']]
                p_rep[name] = [stats['p_rep']]
            else:
                num_examples[name].append(stats['num_examples'])
                is_rep[name].append(stats['is_rep'])
                p_rep[name].append(stats['p_rep'])

    ind = np.arange(len(names))
    num_examples = [num_examples[name] for name in names]
    is_rep = [1.0*sum(is_rep[name])/len(num_examples[0]) for name in names]
    p_rep = [np.mean(p_rep[name]) for name in names]

    avg_sizes = [np.mean(nums) for nums in num_examples]

    width = 0.35
    print avg_sizes
    plt.bar(ind, avg_sizes, width)
    plt.xticks(ind, names)
    plt.title('Average Percentage of Examples Used')
    plt.show()

    plt.boxplot(num_examples, labels=names)
    plt.title('Distribution of Percentage of Examples Used')
    plt.show()

    plt.bar(ind, is_rep, width)
    plt.xticks(ind, names)
    plt.title('Percent Is Completely Representative Set')
    plt.show()

    print p_rep, names
    plt.bar(ind, p_rep, width)
    plt.xticks(ind, names)
    plt.title('Percent of Set Represented')
    plt.show()


if __name__ == '__main__':
    graph_stuff()

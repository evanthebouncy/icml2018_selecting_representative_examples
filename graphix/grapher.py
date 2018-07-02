import pickle
import matplotlib.pyplot as plt
import numpy as np


def gen_graphs(TEST_LOC):
    data = pickle.load( open( TEST_LOC, "rb" ) )

    build_times = []
    solve_times = []
    nn_times = []
    check_times = []
    num_examples = []
    orig_examples = []

    build_times_all = []
    solve_times_all = []
    total_times = []
    
    def change_name(a_name):
      return 'handcraft' if a_name == 'h1+cegis' else a_name

    names = [change_name ( d[0]['method'] ) for d in data]

    for method in data:
        builds = [d['building_time'] for d in method]
        build_times_all.append(builds)
        solves = [d['solving_time'] for d in method]
        solve_times_all.append(solves)
        try:
            nns = [d['nn_time'] for d in method]
        except:
            nns = [0 for d in method]

        try:
            checks = [d['checking_time'] for d in method]
        except:
            checks = [0 for d in method]

        total_time = [builds[i] + solves[i] + nns[i] for i in xrange(len(builds))]
        total_times.append(total_time)

        numexs = [d['ce_size'] for d in method]
        if method[0]['method'] != 'full':
            orig_size = [d['orig_subset_size'] for d in method]
        else:
            orig_size = numexs
        

        nn_times.append(np.mean(nns))        
        build_times.append(np.mean(builds))
        solve_times.append(np.mean(solves))
        orig_examples.append(np.mean(orig_size))
        num_examples.append(np.mean(numexs))


    names2 = names
    names = []
    for name in names2:
        if name == 'nn+cegis':
            names.append('ours')
        else:
            names.append(name)

    width = 0.35
    figsize = (4,3)
    plt.figure(figsize=figsize)
    ind = np.arange(len(data))
    nn_times = np.array(nn_times)
    build_times = np.array(build_times)
    solve_times = np.array(solve_times)
    p0 = plt.barh(ind, nn_times, width, color='#000000')
    p1 = plt.barh(ind, build_times, width, color='#bfbfbf', left=nn_times, hatch='//')
    p2 = plt.barh(ind, solve_times, width, color= '#a6a6a6', left=build_times+nn_times)
    plt.yticks(ind, names)
    # plt.title('Average Total Time Taken')
    plt.legend((p0[0], p1[0], p2[0]), ('NN', 'Build', 'Solve'))

    plt.tight_layout()
    plt.show()

    width = 0.35
    ind = np.arange(len(data))
    plt.figure(figsize=figsize)
    p1 = plt.barh(ind, orig_examples, width, color='#bfbfbf')
    p2 = plt.barh(ind, np.array(num_examples)-np.array(orig_examples), width, color= '#000000', left=orig_examples)
    plt.yticks(ind, names)
    plt.legend((p1[0], p2[0]), ('Base', 'CEGIS'))
    # plt.title('Avg Number of Examples Used')
    plt.tight_layout()
    plt.show()

    # ignores outliers since ruins plot
    fig1, ax1 = plt.subplots(figsize=(8,3))
    box = plt.boxplot(total_times, labels=names, showfliers=True, vert=False)
    # plt.title('Distribution of Total Times')
    plt.setp(box['medians'], color='black')
    ax1.set_xscale('log') 
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    gen_graphs('result.p')

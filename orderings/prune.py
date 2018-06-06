from solver import *

num_nodes = 10

class Node(object):
    
    def __init__(self, name):
        self.name = name
        self.targets = {}

    def add_edge(self, target, index):
        '''keep track of index so that we can delete it later if needed'''
        self.targets[target] = index


class OrderingGraph(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.nodes = [Node(i) for i in xrange(num_nodes)]
        self.examples = []

    def add_examples(self, examples):
        self.examples = examples
        for i, example in enumerate(examples):
            nums, truth = example
            num1, num2 = nums
            if num1 == num2:
                continue
            elif truth:
                self.nodes[num1].add_edge(num2, i)
            else:
                self.nodes[num2].add_edge(num1, i)

    def prune_examples(self):
        all_paths = [] # tuples of (longest path, path)
        for i in xrange(len(self.nodes)):
            for j in xrange(len(self.nodes)):
                if i == j:
                    continue

                # TODO all paths from i to j
                paths = self.find_paths(i, j)
                if len(paths) == 0:
                    continue
                paths = [(len(path), path) for path in paths]

                # sort in reverse order
                paths = sorted(paths, key=lambda x: -x[0])
                all_paths.append(paths[0])

        indices_to_keep = set()
        for (l, path) in all_paths:
            for i in xrange(len(path)-1):
                node = self.nodes[path[i]]
                indices_to_keep.add(node.targets[path[i+1]])

        self.examples = map(lambda x: self.examples[x], indices_to_keep)

    def find_smallest_set(self, examples):
        self.add_examples(examples)
        self.prune_examples()
        examples = self.examples
        self.reset()
        return examples


    def find_paths(self, start, end):
        to_explore = [[start]]
        paths = []

        while len(to_explore) > 0:
            # print to_explore
            partial_path = to_explore.pop()
            last_node = self.nodes[partial_path[-1]]
            # print last_node.targets.keys()
            for node_name in last_node.targets.keys():
                longer_path = partial_path + [node_name]
                if node_name == end:
                    paths.append(longer_path)
                else:
                    to_explore.append(longer_path)
        return paths


def check_representative(small, big):
    '''checks if the smaller set of examples is representative of the bigger set'''
    missing = []
    small = set(small)
    for example in big:
        if not example in small:
            missing.append(example)
    s = OrderSolver()
    for example in small:
        s.add_example(example)

    for example in missing:
        if s.check_ambiguous(example):
            # print example
            return False
    return True


if __name__ == '__main__':
    from gen import *
    ordering = gen_ordering()
    print ordering

    all_examples = get_all_data(ordering)
    sample_examples = get_data(ordering)
    o = OrderingGraph()
    # smallest_possible = o.find_smallest_set(all_examples)
    # assert len(smallest_possible) == 9
    # is_rep = check_representative(smallest_possible, all_examples)
    # print 'smallest possible is representative:', is_rep
    # assert is_rep

    largest_size = 9
    sizes = []
    for _ in xrange(100):
        sample_examples = get_data(ordering)
        smaller = o.find_smallest_set(sample_examples)
        print len(sample_examples), len(smaller)
        # print smaller
        is_rep = check_representative(smaller, sample_examples)
        print 'smallest of sample is representative:', is_rep
        assert is_rep
        sizes.append(len(smaller))
    print 'largest size subset found:', max(sizes)
    print 'average:', 1.*sum(sizes)/len(sizes)

    # import matplotlib.pyplot as plt
    # plt.title('Sizes of Subsets')
    # plt.boxplot(sizes)
    # plt.show()

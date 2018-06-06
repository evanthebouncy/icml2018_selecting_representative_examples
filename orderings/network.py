import networkx as nx
# import matplotlib.pyplot as plt
import pygraphviz as pgv
from gen import *
from solver import *
from prune import *
from model import *
import pickle
import random
import tqdm


# g = nx.DiGraph()
# # g.add_node(1)
# # g.add_node(2)
# # g.add_node(3)
# # g.add_node(4)
# # g.add_node(5)
# nodes = [1, 2, 3, 4, 5][::-1]
# g.add_nodes_from(nodes)

# # g.add_edges_from([(4, 5), (2, 5), (3, 4), (1, 3)])

# # plt.subplot(1)
# # nx.draw(g, with_labels=True)
# # plt.show()
# # nx.drawing.nx_agraph.write_dot(g,'graph.dot')
# G = nx.drawing.nx_agraph.to_agraph(g)

# g = pgv.AGraph(directed=True,strict=True,rankdir='BT')
# g.add_edge(1, 3)
# g.add_edge(3, 4)
# g.add_edge(4, 5)
# g.add_edge(5, 4, arrowhead='inv')
# g.add_edge(2, 5)
# g.add_edge(3, 3)
# g.add_edge(4, 4)
# g.add_edge(1, 5)


# # G = pgv.AGraph("graph.dot")
# g.layout(prog='dot')
# g.draw('graph.png')


def construct_graph(name, examples):
    g = pgv.AGraph(directed=True, rankdir='BT', overlap='scalexy')
    g.graph_attr['splines']='true'
    # g.graph_attr['nodesep']=1
    # g.layout(prog='dot')
    nodes = [8, 9, 7, 0, 1, 5, 2, 6, 4, 3]
    # g.add_nodes_from(nodes)
    num = 0
    for node in nodes:
        pos = "{},10!".format(num, num)
        g.add_node(node, pos=pos, width='0.5')
        num += 1.0

    for example in examples:
        nums, is_less = example
        num1, num2 = nums
        if num1 == num2:
            g.add_edge(num1, num2, arrowhead='inv')

        if is_less:
            g.add_edge(num1, num2)
        else:
            g.add_edge(num1, num2, arrowhead='inv')
    g.layout(prog='neato')

    g.draw('{}.png'.format(name))


def make_graphs():
    TEST_LOC = "./data/data_test.p"
    MODEL_LOC = './models/m1/oracle.ckpt'

    # breaks when instatiate twice...
    oracle = Oracle('oracle')
    oracle.restore_model(MODEL_LOC)

    data = pickle.load( open( TEST_LOC, "rb" ) )

    for i, sample_examples in enumerate(data):
        if i != 7:
            continue
        
        # random.shuffle(sample_examples)
        random_examples = sample_examples[:int(len(sample_examples)*.35)]
        random_rep = check_representative(random_examples, sample_examples)

        o = OrderingGraph()
        hasse_examples = o.find_smallest_set(sample_examples)        

        c = OrderCEGIS()
        cegis_examples = c.solve(sample_examples)
        cegis_rep = check_representative(cegis_examples, sample_examples)

        oracle_examples = oracle.get_until_confident(sample_examples, confidence=0.75)
        nn_rep = check_representative(oracle_examples, sample_examples)

        if nn_rep and not random_rep and not cegis_rep:
            print len(sample_examples), i
            construct_graph('all', sample_examples)
            construct_graph('random', random_examples)
            construct_graph('hasse', hasse_examples)
            construct_graph('cegis', cegis_examples)
            construct_graph('nn', oracle_examples)
            quit()
            # raw_input("Press Enter to continue...")

if __name__ == '__main__':
    make_graphs()
    # 7 is good










from gen import *
from solver import *
from prune import *
from model import *
import pickle
import random
import tqdm

TEST_LOC = "./data/data_test.p"
MODEL_LOC = './models/m1/oracle.ckpt'

# breaks when instatiate twice...
oracle = Oracle('oracle')
oracle.restore_model(MODEL_LOC)


def trial(sample_examples):
    stats = {}

    random_examples = sample_examples[:int(len(sample_examples)*.35)]
    is_rep = check_representative(random_examples, sample_examples)
    p_rep = percent_representative(random_examples, sample_examples)
    stats['random35'] = {'num_examples': 1.0*len(random_examples)/len(sample_examples), 'is_rep': is_rep, 'p_rep': p_rep}

    # do random first
    random_examples = sample_examples[:int(len(sample_examples)*.80)]
    is_rep = check_representative(random_examples, sample_examples)
    p_rep = percent_representative(random_examples, sample_examples)
    stats['random80'] = {'num_examples': 1.0*len(random_examples)/len(sample_examples), 'is_rep': is_rep, 'p_rep': p_rep}

    # hasse set
    o = OrderingGraph()
    hasse_examples = o.find_smallest_set(sample_examples)
    is_rep = check_representative(hasse_examples, sample_examples)
    p_rep = percent_representative(hasse_examples, sample_examples)
    stats['hasse'] = {'num_examples': 1.0*len(hasse_examples)/len(sample_examples), 'is_rep': is_rep, 'p_rep': p_rep}

    # cegis set
    c = OrderCEGIS()
    cegis_examples = c.solve(sample_examples)
    is_rep = check_representative(cegis_examples, sample_examples)
    p_rep = percent_representative(cegis_examples, sample_examples)
    stats['cegis'] = {'num_examples': 1.0*len(cegis_examples)/len(sample_examples), 'is_rep': is_rep, 'p_rep': p_rep}

    # oracle is global variable
    oracle_examples = oracle.get_until_confident(sample_examples, confidence=0.9)
    is_rep = check_representative(oracle_examples, sample_examples)
    p_rep = percent_representative(oracle_examples, sample_examples)
    stats['nn'] = {'num_examples': 1.0*len(oracle_examples)/len(sample_examples), 'is_rep': is_rep, 'p_rep': p_rep}
    return stats


def percent_representative(small, big):
    '''checks if the smaller set of examples is representative of the bigger set'''
    missing = []
    small = set(small)
    for example in big:
        if not example in small:
            missing.append(example)
    s = OrderSolver()
    for example in small:
        s.add_example(example)

    wrong = 0.

    for example in missing:
        if s.check_ambiguous(example):
            # print example
            wrong += 1

    return 1-wrong/len(big)



def generate_data(n=500):
    to_write = []
    for i in xrange(n):
        ordering = gen_ordering()
        sample_examples = get_data(ordering)
        random.shuffle(sample_examples)
        to_write.append(sample_examples)

    pickle.dump( to_write, open( TEST_LOC, "wb" ) )

def load_data():
    return pickle.load( open( TEST_LOC, "rb" ) )


if __name__ == '__main__':
    generate_data()
    data = load_data()
    stats = []

    for examples in tqdm.tqdm(data):
        stats.append(trial(examples))

    pickle.dump( (stats), 
                   open( "results.p", "wb" ) )

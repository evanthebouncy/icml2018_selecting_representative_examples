from model import *
import sys
from dfa_solver import *
from gen import *
import random
import time
import hashlib
import time
from invert import *
import pickle
TEST_LOC = "./data/data_test.p"

def make_test_data(n=400):
  to_write = []
  for i in range(n):
    test_mat = sample_matrix()
    all_data = generate_examples(test_mat, 2000)
    all_data = dedup(all_data)
    print i, " ", len(all_data)
    to_write.append(all_data)
  pickle.dump( to_write, open( TEST_LOC, "wb" ) )

def data_ratio(data):
  pos, neg = 0, 0
  for x in data:
    if x[1] is True:
      pos += 1
    else:
      neg += 1
  return pos, neg


if __name__ == "__main__":
#  make_test_data()
#  assert 0
  # store_loc = "_time_exp_result.p"
  store_loc = "_time_exp_rand_only.p"

  invert = Inverter('./models/m1/oracle.ckpt')
  test_data = pickle.load( open( TEST_LOC, "rb" ) )

  results = []
  
  for idx, all_data in enumerate(test_data):
    print all_data[:100]

    print "testing iteration ", idx, " balance ", data_ratio(all_data)
    answers = [
        #              invert.invert_full(all_data, "full"),
        #              invert.invert_cegis(all_data, [], "cegis"),
              invert.invert_full(all_data, "rand+cegis", fraction=0.3),
              invert.invert_full(all_data, "rand+cegis", fraction=0.5),
              invert.invert_full(all_data, "rand+cegis", fraction=0.7),
              #              invert.invert_full(all_data, "h1+cegis"),
              # invert.invert_full(all_data, "nn+cegis", confidence=0.6),
              ]
    if len(results) == 0:
      results = [[] for _ in range(len(answers))]
    assert len(answers) == len(results)
    for i in range(len(results)):
      results[i].append(answers[i])

    if idx % 2 == 0:
      print "dumping pickle "
      pickle.dump( results, open( store_loc, "wb" ) )

  print "dumping pickle "
  pickle.dump( results, open( store_loc, "wb" ) )


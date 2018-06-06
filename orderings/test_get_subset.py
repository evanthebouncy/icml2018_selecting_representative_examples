from model import *
from gen import *
MODEL_LOC = './models/m1/oracle.ckpt'
from solver import *
from prune import *

if __name__ == "__main__":
  oracle = Oracle("oracle")
  oracle.restore_model(MODEL_LOC)

  ord1 = gen_ordering()
  data = get_data(ord1)
  print ord1
  print data

  data_subset = oracle.get_until_confident(data, confidence=0.9)
  print data_subset

  print "wolololo"
  print len(data)
  for d in data:
    print d

  print len(data_subset)
  for d in data_subset:
    print d

  is_rep = check_representative(data_subset, data)
  print len(data), len(data_subset), is_rep

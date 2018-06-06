from model import *
import sys
from dfa_solver import *
from gen import *
from h1 import *
import random
import time
import hashlib
import time

# convert numpy back into data form
def np_to_data(np_data):
  ret = []
  for nd in np_data:
    x, y = nd
    str1 = []
    for xx in x:
      str1.append(int(np.argmax(xx)))
    out = True if y[0] > y[1] else False
    ret.append( (str1, out) )
  return ret

# check if something is solved, if not return counter examples
def check_solved(synth, all_pairs):
  wrong = []
  for x,y in all_pairs:
    if accept_state(execute_dfa(synth, x)) != y:
      wrong.append((x,y))
  return wrong


def choose_h1(data):
  # try to get as diverse of endings as possible
  stuffs = [(i, (''.join(map(str, x[0][::-1]))), x[1]) for i, x in enumerate(data)]

  tree = H1Tree()
  # print
  # print sorted(data1, key=lambda x: x[1])
  for (index, s, yes) in stuffs:
    tree.add_example(s, yes, index)

  example_indices = tree.get_top_examples()
  data1 = [data[i] for i in example_indices]
  return data1

class Inverter:

  def __init__(self, model_loc):
    self.oracle = Oracle("oracle")
    self.oracle.restore_model(model_loc)
    self.clear_solver()

  def clear_solver(self):
    self.s = DFA_Solver(N_STATES, N_CHAR)

  def pick_arbitrary(self, ces):
    ces_ordered = [(int(hashlib.sha1(str(x)).hexdigest(),16),x) for x in ces]
    return min(ces_ordered)[1]

  def invert_cegis(self, all_data, d_subset, method):
    self.clear_solver()
    # build time, solve time, check time
    b_time = 0.0
    s_time = 0.0
    c_time = 0.0
    # print method
    assert method in ["cegis", "r_cegis", 'a_cegis',]
    data_subset = all_data[:1] if len(d_subset) == 0 else [x for x in d_subset]

    # ces is stored in idx space
    # pick a counter example, also return in idx space
    def ce_picker(ces):
      if method == "cegis":
        return ces[0]
      if method == "r_cegis":
        return random.choice(ces)
      if method == "a_cegis":
        return self.pick_arbitrary(ces)

    # add initial data subset
    b_start = time.time()
    print "adding . . ."
    for i, x_y in enumerate(data_subset):
      x,y = x_y
      self.s.add_example(i, x, y)
    b_time += time.time() - b_start

    i = len(data_subset)
    # do the cegis
    while True:
      solve_start = time.time()
      print "solving . . . ", i
      synth = self.s.get_matrix()
      s_time += time.time()-solve_start
      check_start = time.time()
      cex = check_solved(synth, all_data)
      c_time += time.time()-check_start
      if len(cex) == 0:
        return {"method" : method,
                "build_time" : b_time,
                "solve_time" : s_time,
                "n_examples" : len(data_subset),
                "correct"    : True,
                "check_time" : c_time,
               }
      else:
        ce = ce_picker(cex)
        data_subset.append(ce)
        self.s.add_example(i, ce[0], ce[1])
      i += 1


  def invert(self, all_data, method, confidence=0.9, fraction=0.3):
    assert method in ["full", "rand", "nn", "nn+cegis", "rand+cegis", "nn_experiment", 'h1+cegis']

    if method == "full":
      self.clear_solver()
      ret = dict()
      build_start = time.time()
      print "[full] adding . . ."
      for i, x_y in enumerate(all_data):
        x,y = x_y
        self.s.add_example(i, x, y)
      build_time = time.time()-build_start
      solve_start = time.time()
      print "[full] solving . . . "
      synth = self.s.get_matrix()
      solve_time = time.time()-solve_start
      correct = len(check_solved(synth, all_data)) == 0
      return {'method' : method,
              'build_time' : build_time,
              'solve_time' : solve_time,
              'n_examples' : len(all_data),
              'correct'    : correct,
              }

    if method == "nn":
      self.clear_solver()
      ret = dict()

      nn_start = time.time()
      sub_data = self.oracle.get_until_confident(all_data, confidence=confidence)
      nn_time = time.time()-nn_start

      build_start = time.time()
      print len(sub_data)
      sub_data = np_to_data(sub_data)
      print "[nn] adding . . ."
      for i, x_y in enumerate(sub_data):
        x,y = x_y
        self.s.add_example(i, x, y)
      build_time = time.time()-build_start
      solve_start = time.time()
      print "[nn] solving . . . ", len(sub_data)
      synth = self.s.get_matrix()
      solve_time = time.time()-solve_start
      correct = len(check_solved(synth, all_data)) == 0
      return {'method' : method,
              'nn_time'    : nn_time,
              'build_time' : build_time,
              'solve_time' : solve_time,
              'n_examples' : len(sub_data),
              'correct'    : correct,
              }

    if method == "rand+cegis":
      print "[rand+cegis] . . . "
      print fraction
      self.clear_solver()
      ret = dict()
      sub_data = all_data[:int(fraction * len(all_data) )] 
      orig_len = len(sub_data)

      to_ret = self.invert_cegis(all_data, sub_data, "cegis")
      to_ret['method'] = method
      to_ret['n_examples_orig'] = orig_len
      return to_ret

    if method == "nn+cegis":

      nn_start = time.time()
      sub_data = self.oracle.get_until_confident(all_data, confidence=confidence)
      nn_time = time.time()-nn_start

      # sub_data = np_to_data(sub_data)
      orig_len = len(sub_data)
      print "[nn+cegis] . . . ", orig_len
      to_ret = self.invert_cegis(all_data, sub_data, "cegis")
      to_ret['method'] = method
      to_ret['n_examples_orig'] = orig_len
      to_ret['nn_time'] = nn_time
      return to_ret

    if method == "nn_experiment":
      s_time = time.time()
      trace_obs = self.oracle.get_trace(full_img, 20, confidence)
      nn_time = time.time() - s_time
      sub_constraints = obs_to_constraints(trace_obs, full_img)

      params = self.s.solve(8, sub_constraints)
      square,line = mk_scene(params)
      recovered = render(square+line)
      diff = full_img - recovered
      diff_idx1, diff_idx2 = np.where(diff != 0)
      self.clear_solver()
      orig_size = len(sub_constraints)

      params = self.invert_cegis(constraints, full_img, "r_cegis", sub_constraints)
      params['nn_time'] = nn_time
      params['error'] = float(len(diff_idx1))
      params['orig_subset_size'] = orig_size
      return params

    if method == 'h1+cegis':
      print "[h1+cegis] . . . "
      self.clear_solver()
      ret = dict()
      sub_data = choose_h1(all_data)
      orig_len = len(sub_data)

      to_ret = self.invert_cegis(all_data, sub_data, "cegis")
      to_ret['method'] = method
      to_ret['n_examples_orig'] = orig_len
      return to_ret

if __name__ == "__main__":
  invert = Inverter('./models/m1/oracle.ckpt')

  test_mat = sample_matrix()
  all_data = generate_examples(test_mat, 500)

  all_data = dedup(all_data)
  print len(all_data)

  print all_data[:20]

  #inv_full_ans = invert.invert_full(all_data, "full")
  inv_rand_ans = invert.invert_full(all_data, "rand+cegis")
  inv_h1_ans = invert.invert_full(all_data, "h1+cegis")
  #inv_nn_ans = invert.invert_full(all_data, "nn", confidence=0.8)
  inv_nn_cegis_ans = invert.invert_full(all_data, "nn+cegis", confidence=0.9)
  #inv_cegis_ans = invert.invert_cegis(all_data, [], 'cegis')

  #print inv_full_ans
  print inv_rand_ans
  print inv_h1_ans
  #print inv_nn_ans
  print inv_nn_cegis_ans
  #print inv_cegis_ans



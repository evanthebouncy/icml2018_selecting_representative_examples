import random
from random import randint
import numpy as np

# N_CHAR = 2
# L = 10
# N_STATES = 6

N_CHAR = 2
L = 10
N_STATES = 6

N_CLOSE = 10

def get_letter():
  return randint(0, N_CHAR-1)

def get_input_string(L):
  return [get_letter() for _ in range(L)]

def get_example_dist(e1, e2):
  return get_str_dist(e1[0], e2[0])

# return a negative number of distance, smaller the number (more neg) the closer
def get_str_dist(str_a, str_b):
  def get_str_dist_forward(str_a, str_b):
    backward_cnt = 0
    for i in range(1, len(str_a)+1):
      if str_a[i-1] != str_b[i-1]:
        return - backward_cnt
      else:
        backward_cnt += 1
    return -backward_cnt
  def get_str_dist_backward(str_a, str_b):
    backward_cnt = 0
    for i in range(1, len(str_a)+1):
      if str_a[-i] != str_b[-i]:
        return - backward_cnt
      else:
        backward_cnt += 1
    return -backward_cnt
  return get_str_dist_forward(str_a, str_b), get_str_dist_backward(str_a, str_b)

# get N_CLOSE for both forward and backward closeness
# from the input_data to the to_decide
def get_close(input_data, to_decide):
  forward_dists = sorted( [(get_example_dist(x, to_decide)[0], x) for x in input_data] )
  backward_dists = sorted( [(get_example_dist(x, to_decide)[1], x) for x in input_data] )
  return forward_dists[:N_CLOSE], backward_dists[:N_CLOSE]


# a matrix is valid only if it can accept some and reject some other
def check_valid(matrix):
  pos, neg = [], []
  for i in range(100):
    input_str = get_input_string(L)
    output_TF = accept_state( execute_dfa(matrix, input_str) )

    if output_TF is True:
      pos.append((input_str, output_TF))
    if output_TF is False:
      neg.append((input_str, output_TF))
  return len(pos) > 0 and len(neg) > 0

# the transition matrix is a matrix of shape [N_STATES, N_CHAR] and each entry N_STATE
def sample_matrix():
  ret = [[randint(0, N_STATES-1) for i in range(N_CHAR)] for _ in range(N_STATES)]
  if check_valid(ret):
    return ret
  else:
    return sample_matrix()

# to use the transition matrix, notice it is of shape
# [N_STATES, N_CHAR] and each entry N_STATE
# so if cur_state is 3, and you read character 2, you go to
# transition_matrix[3][2] and read off the value there, say it's 1
# now cur_state is 1
def execute_dfa(matrix, input_string):
  cur_state = 0
  # print matrix
  try:
    for x in input_string:
      cur_state = matrix[cur_state][x]
    # print x, cur_state
    return cur_state
  except:
    # might get IndexError: string index out of range if doesn't exist, so clearly wrong
    return -1

# we deem a dfa "accepts" an input string if the final state is the last one
def accept_state(state):
  return state == (N_STATES - 1)

def dedup(data):
  ahem = set()
  for d in data:
    ahem.add(repr(d))
  ret = []
  for d in ahem:
    ret.append(eval(d))
  return ret


# generate n input output examples
def generate_examples(matrix, n):
  return generate_balanced_examples(matrix, n)
  ret = []

  while len(ret) < n:
    input_str = get_input_string(L)
    output_TF = accept_state( execute_dfa(matrix, input_str) )
    ret.append( (input_str, output_TF) )

  random.shuffle(ret)
  return ret

# generate n input output examples
def generate_balanced_examples(matrix, n):
  pos = []
  neg = []

  while min( len(pos), len(neg) ) < n / 2:
    input_str = get_input_string(L)
    output_TF = accept_state( execute_dfa(matrix, input_str) )

    if output_TF is True:
      pos.append((input_str, output_TF))
    if output_TF is False:
      neg.append((input_str, output_TF))

  ret = pos[:n/2] + neg[:n/2]
  random.shuffle(ret)
  return ret


# turn examples into numpy arrays
def examples_to_numpy(examples):
  def to_1hot(char):
    to_ret = [0.0 for _ in range(N_CHAR)]
    to_ret[char] = 1.0
    return to_ret

  ret_in, ret_out = [], []
  for e in examples:
    xx, outt = e
    xx_np = [to_1hot(x) for x in xx]
    outt_np = [1.0, 0.0] if outt else [0.0, 1.0]
    ret_in.append(xx_np)
    ret_out.append(outt_np)

  return np.array(ret_in), np.array(ret_out)

def gen_skewed_number(n=2000):
  ret = []
  bnd = int (n / N_CLOSE)
  for i in range(bnd):
    for j in range(i):
      ret.append((bnd-i) * N_CLOSE)
  return ret + [1000]


def gen_train_data(n=200):
  m = sample_matrix()

  skew_sizes = gen_skewed_number(n)
  input_sample_size = random.choice(skew_sizes)+2
  # print input_sample_size

  examples = dedup(generate_balanced_examples(m, input_sample_size))
  # print len(examples)

  to_be_decided = examples[-1]
  # print to_be_decided
  data_input = examples[:-1]
  if len(data_input) < 11:
    return gen_train_data(n)
  front, back = get_close(data_input, to_be_decided)
  return [x[1] for x in front], [x[1] for x in back], to_be_decided

def data_to_numpy(front, back, to_decide):
  front_strs, front_TFs = examples_to_numpy(front)
  back_strs, back_TFs = examples_to_numpy(back)
  to_decide_strs, to_decide_TFs = examples_to_numpy([to_decide])
  return front_strs, front_TFs, back_strs, back_TFs, to_decide_strs[0], to_decide_TFs[0]

def gen_batch_data(n_batch=20):
  b_front_strs, b_front_TFs, b_back_strs, b_back_TFs, b_to_decide_strs, b_to_decide_TFs = [],[],[],[],[],[]
  for i in range(n_batch):
    front_i, back_i, to_decide_i = gen_train_data()
    front_strs, front_TFs, back_strs, back_TFs, to_decide_strs, to_decide_TFs = data_to_numpy(front_i, back_i, to_decide_i)
    b_front_strs.append(front_strs)
    b_front_TFs.append(front_TFs)
    b_back_strs.append(back_strs)
    b_back_TFs.append(back_TFs)
    b_to_decide_strs.append(to_decide_strs)
    b_to_decide_TFs.append(to_decide_TFs)
  return np.array(b_front_strs), np.array(b_front_TFs), np.array(b_back_strs), np.array(b_back_TFs), np.array(b_to_decide_strs), np.array(b_to_decide_TFs)

#
#  all_idxs = [i for i in range(n)]
#  random.shuffle(all_idxs)
#
#  observed_idxs   = all_idxs[:to_observe]
#  unobserved_idxs = all_idxs[to_observe:]
#
#  seen_in, seen_out = [np_in[i] for i in observed_idxs], [np_out[i] for i in observed_idxs]
#  unseen_in, unseen_out = [np_in[i] for i in unobserved_idxs], [np_out[i] for i in unobserved_idxs]
#
#  if not together:
#    return np.array(seen_in),\
#           np.array(seen_out),\
#           np.array(unseen_in),\
#           np.array(unseen_out)
#  else:
#    return np_in, np_out

def gen_exam():
  m = sample_matrix()
  examples = generate_examples(m, 100)
  train = examples[:80]
  test = examples[80:]
  return train, test

############### testing ##########
def test_dist():
  same,diff = 0,0
  for _ in range(100):
    m = sample_matrix()
    examples = generate_examples(m, 100)
    examples = dedup(examples)

    chosen = examples[0]
    # [0] is forward, [1] is backward distance
    haha = [(get_example_dist(e, chosen)[1], e) for e in examples]
    haha= sorted(haha)
    for ha in haha[:5]:
      if ha[1][1] == chosen[1]:
        same += 1
      else:
        diff += 1
  print same, " ", diff

def test_close_data():
  forward, backward, to_decide = gen_train_data()
  print "forward "
  for f in forward:
    print f
  print "backward "
  for b in backward:
    print b
  print "to decide "
  print to_decide

  b_f_s, b_f_tf, b_b_s, b_b_tf, b_dec_s, b_dec_tf = gen_batch_data(20)
  print b_f_s.shape
  print b_f_tf.shape
  print b_b_s.shape
  print b_b_tf.shape
  print b_dec_s.shape
  print b_dec_tf.shape
  

if __name__ == "__main__":
  # test_dist()
  test_close_data()



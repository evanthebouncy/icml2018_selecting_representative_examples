# the dfa solver: given a set of input-output pairs
# in the form of sequence -> true/false
# solve for the dfa that is consistent with those pairs

from z3 import *

class DFA_Solver:
  # number of states, number of characters
  # optionally a transition matrix so we can verify the forward function
  def __init__(self, n_states=5, n_chars=3, T=None):  
    self.solver = Solver()

    # 10 minute timeout
    self.solver.set("timeout", 10*60*1000)

    self.n_states = n_states
    self.n_chars = n_chars

    # regardless, we initialize a transition matrix
    self.T = []
    for i in range(self.n_states):
      to_add_row = [Int('T|{}_{}'.format(i, j)) for j in range(self.n_chars)]
      self.T.append(to_add_row)
      # bound their range
      for xx in to_add_row:
        self.solver.add(xx >= 0)
        self.solver.add(xx < self.n_states)
        
    if T != None:
      print "initializing a fixed transition matrix, for forward-execution debugging only"
      print "for synthesis do not initialize this T"
      assert len(T) == self.n_states
      assert len(T[0]) == self.n_chars
      for i_state in range(self.n_states):
        for j_char in range(self.n_chars):
          self.solver.add(self.T[i_state][j_char] == T[i_state][j_char])

  def get_model(self):
    if self.solver.check() == sat:
      model = self.solver.model()
      return model
    else:
      return "UNSAT"

  def get_matrix(self):
    if self.solver.check() == sat:
      model = self.solver.model()
      matrix = [[0 for i in xrange(len(self.T[0]))] for j in xrange(len(self.T))]

      for i in xrange(len(self.T)):
        for j in xrange(len(self.T[0])):
          matrix[i][j] = model[self.T[i][j]].as_long()

      return matrix
    else:
      return "UNSAT"

  # starting with initial state = 0, simulate the dfa execution on input string
  # k denotes the k-th forward execution, different k value creates fresh variable instances
  def forward_execution(self, k, input_string):
    cur_state = Int("s|{}_{}".format(k, -1))
    self.solver.add(cur_state == 0)
    for idx, x in enumerate(input_string):
      new_state = Int("s|{}_{}".format(k, idx))

      new_value = cur_state
      # loop through all possible transitions . . .
      # the most hardest part of the logic
      for i_state in range(self.n_states):
        for j_char in range(self.n_chars):
          state_match = cur_state == i_state
          char_match  = x == j_char
          new_value = If(And(state_match, char_match), self.T[i_state][j_char], new_value)

      self.solver.add(new_state == new_value)
      cur_state = new_state

    # return the last state
    return cur_state

  def add_example(self, k, input_string, acceptance):
    last_state = self.forward_execution(k, input_string)
    is_accept = last_state == (self.n_states - 1)
    self.solver.add(is_accept == acceptance)
          
if __name__ == "__main__":
  from gen import *
  import time

  # for i in range(10):
  #   print "checking forward execution ", i
  #   T_sample = sample_matrix()
  #   sample_str = get_input_string(10)
  #   print T_sample
  #   print sample_str
  #   last_state_gen_py = execute_dfa(T_sample, sample_str)

  #   dfa_solver = DFA_Solver(T=T_sample)
  #   last_state = dfa_solver.forward_execution(1, sample_str)
  #   model = dfa_solver.get_model()
  #   last_state_solver = model[last_state].as_long()
  #   print "executing sample_str on T_sample produced [python] ", last_state_gen_py, " [solver]", last_state_solver
  #   assert last_state_gen_py == last_state_solver, "python and solver didnt match cyka debil"

  for i in range(10):
    print "checking maybe ? synthesis ", i
    T_sample = sample_matrix()
    print T_sample
    examples = generate_examples(T_sample, 5000)
    # we keep the matrix implicit here with T=None, so hopefully solver will solve it
    dfa_solver = DFA_Solver(T=None)
    start = time.time()
    print "adding {} examples . . .".format(len(examples))
    for idx, e in enumerate(examples):
      input_str, accept = e
      dfa_solver.add_example(idx, input_str, accept)
    print 'took', time.time()-start, 'seconds'
    start = time.time()
    print "solving . . . "

    solved_model = dfa_solver.get_model()
    # print solved_model
    print dfa_solver.get_matrix()
    assert solved_model != "UNSAT", "you got unsat urod blyat!"
    print 'took', time.time()-start, 'seconds'



 

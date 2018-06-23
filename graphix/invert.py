from graphix_solver import DrawingSolver, check
from model import *
from draw import *
import sys
import random
import time
import hashlib

def obs_to_constraints(trace, render):
  trace_coords = [x[0] for x in trace]

  sub_constraints = []
  for cc in trace_coords:
    # flip the coordinate to x,y space
    id1,id2 = cc
    x,y = rev(id1,id2)
    val = bool(render[id1][id2] == 1.0)
    sub_constraints.append(((x,y),val))
  return sub_constraints

def h1(constraints):
  dick = dict(constraints)
  # return a list of neighbors' values of data in dictionary d
  def neibor(xy, dick):
    ret = []
    x,y = xy
    for d1 in dick:
      xy1, ans1 = d1, dick[d1]
      x1,y1 = xy1
      if abs(x1 - x) <= 1 and abs(y1-y) <= 1:
        ret.append(ans1)
    return ret

  to_ret = []
  for dd in dick:
    neib = neibor(dd, dick)
    if (True in neib) and (False in neib):
      to_ret.append((dd, dick[dd]))
  return to_ret

class Inverter:

  def __init__(self):
    self.s = DrawingSolver()
    self.impnet = Implynet(tf.Session())
    self.impnet.load_model("./models/m1/imply.ckpt")

  def clear_solver(self):
    self.s = DrawingSolver()

  def pick_arbitrary(self, ces):
    ces_ordered = [(int(hashlib.sha1(str(x)).hexdigest(),16),x) for x in ces]
    return min(ces_ordered)[1]

  def invert_cegis(self, constraints, render, method="cegis", sub_constraints=None):
    self.clear_solver()
    b_time = 0.0
    s_time = 0.0
    c_time = 0.0
    # print method
    assert method in ["cegis", "r_cegis", 'a_cegis',]
    i = 0
    sub_constraints = constraints[:1] if sub_constraints == None else sub_constraints

    # ces is stored in idx space
    # pick a counter example, also return in idx space
    def ce_picker(ces):
      if method == "cegis":
        return ces[0]
      if method == "r_cegis":
        return random.choice(ces)
      if method == "a_cegis":
        return self.pick_arbitrary(ces)

    while True:
      i += 1
      paras = self.s.solve(8, sub_constraints, {})
      # print paras
      b_time += paras['building_time']
      s_time += paras['solving_time']
      # counter example in idx space
      stime = time.time()
      ces = check(paras, render, i)
      # add check time
      c_time += time.time() - stime
      if ces == None:
        paras['building_time'] = b_time
        paras['solving_time'] = s_time
        paras['checking_time'] = c_time
        paras['ce_size'] = len(sub_constraints)
        paras['method'] = method
        paras['orig_subset_size'] = 0
        return paras
      else:
        id1,id2 = ce_picker(ces)
        x,y = rev(id1,id2)
        val = bool(render[id1][id2] == 1)
        sub_constraints +=    [((int(x), int(y)), val)]

  def invert_full(self, constraints, full_img, method="full", confidence=0.9):
    self.clear_solver()
    fraction = 0.2
    assert method in ["full", "rand", "nn", "nn+cegis", "rand+cegis", "h1+cegis", "nn_experiment"]

    if method == "full":
      params = self.s.solve(8, constraints)
      params['ce_size'] = len(constraints)
      params['method'] = method
      return params

    if method == "rand":
      sub_constraints = random.sample(constraints, int(fraction * len(constraints)))
      params = self.s.solve(8, sub_constraints)
      params['ce_size'] = len(sub_constraints)
      square,line = mk_scene(params)
      recovered = render(square+line)
      diff = full_img - recovered
      diff_idx1, diff_idx2 = np.where(diff != 0)
      params['error'] = float(len(diff_idx1))
      return params

    if method == "nn":
      s_time = time.time()
      trace_obs = self.impnet.get_trace(full_img, 20, confidence)
      nn_time = time.time() - s_time
      sub_constraints = obs_to_constraints(trace_obs, full_img)
      params = self.s.solve(8, sub_constraints)
      params['ce_size'] = len(sub_constraints)
      params['nn_time'] = nn_time
      return params

    if method == "rand+cegis":
      sub_constraints = random.sample(constraints, int(fraction * len(constraints)))
      orig_size = len(sub_constraints)
      params = self.invert_cegis(constraints, full_img, "r_cegis", sub_constraints)
      params['method'] = method
      params['orig_subset_size'] = orig_size
      return params

    if method == "h1+cegis":
      sub_constraints = h1(constraints)
      orig_size = len(sub_constraints)
      params = self.invert_cegis(constraints, full_img, "r_cegis", sub_constraints)
      params['method'] = method
      params['orig_subset_size'] = orig_size
      return params

    if method == "nn+cegis":
      s_time = time.time()
      trace_obs = self.impnet.get_trace(full_img, 20, confidence)
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
      params['method'] = method
      return params

    if method == "nn_experiment":
      s_time = time.time()
      trace_obs = self.impnet.get_trace(full_img, 20, confidence)
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


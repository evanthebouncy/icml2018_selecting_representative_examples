import random
import numpy as np

L = 10

def gen_ordering():
  to_ret = [_ for _ in range(L)]
  random.shuffle(to_ret)
  return to_ret

def get_all_data(ordering):
  to_ret = []
  for i in range(L):
    for j in range(L):
      i_pos = ordering.index(i)
      j_pos = ordering.index(j)
      to_ret.append( ( (i,j), i_pos < j_pos) )
  return to_ret

def get_data(ordering):
  all_data = get_all_data(ordering)
  rand_len = int(len(all_data) * np.random.uniform(low=0.3, high=1.0))
  random.shuffle(all_data)
  return all_data[:rand_len]

def get_train_data(data):
  to_observe = random.randint(0, len(data)-2)
  return data[:to_observe], data[-1]

def observed_to_np(data):
  ret = [ [ [0.0, 0.0] for ii in range(L)] for jj in range(L)]
  for x_y,tf in data:
    x,y = x_y
    if tf:
      ret[x][y] = [1.0, 0.0]
    if not tf:
      ret[x][y] = [0.0, 1.0]
  return np.array(ret)

def query_to_np(query):
  qq, tf = query
  x, y = qq
  xx, yy, ttff = [0.0 for _ in range(L)], [0.0 for _ in range(L)], [1.0, 0.0] if tf else [0.0, 1.0]
  xx[x] = 1.0
  yy[y] = 1.0

  return xx, yy, ttff

def gen_batch_data(n=20):
  b_ob, b_q_x, b_q_y, b_tf = [],[],[],[]
  for i in range(n):
    ord1 = gen_ordering()
    to_ob, to_pred = get_train_data(get_data(ord1))
    ob = observed_to_np(to_ob)
    qx, qy, qtf = query_to_np(to_pred)
    b_ob.append(ob)
    b_q_x.append(qx)
    b_q_y.append(qy)
    b_tf.append(qtf)
  return np.array(b_ob), np.array(b_q_x), np.array(b_q_y), np.array(b_tf)

if __name__ == "__main__":
  ord1 = gen_ordering()
  print ord1
  data = get_data(ord1)
  print data
  print
  print
  train_in, train_out = get_train_data(data)

  print train_in
  print observed_to_np(train_in)
  print train_out
  print query_to_np(train_out)


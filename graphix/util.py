import sys
import numpy as np
from graphix_lang import L

def same_line_print(message):
  sys.stdout.write("\r" + message)
  sys.stdout.flush()

def mk_query(img):
  assert len(img.shape) == 2
  def qry(id1_id2):
    id1,id2 = id1_id2
    if img[id1][id2] == 1.0:
      return [1.0, 0.0]
    else:
      return [0.0, 1.0]
  return qry

def rev(x,y):
  return y,x

# takes an image (with reversed index)
# to x y which is actually ordered id2 id1
def img_2_constraints(img):
  M,N = img.shape
  ret = []
  for id1 in range(M):
    for id2 in range(N):
      val = bool(img[id1][id2] == 1)
      ret.append(((id2,id1),val))
      
  return ret

# turn image into a label, preserve the reverse ordering of index
def img_2_labels(img):
  full_obs = np.zeros([L,L,2])
  qry = mk_query(img)
  for i in range(L):
    for j in range(L):
      full_obs[i][j] = qry((i,j))
  return full_obs


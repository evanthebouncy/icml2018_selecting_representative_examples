import numpy as np
import matplotlib.pylab as plt
import multiprocessing as mp
from graphix_lang import L

from matplotlib import figure

FIG = plt.figure()

def draw_coord(coord, name, lab=[1.0, 0.0]):
  color = 1.0 if lab[0] > lab[1] else -1.0
  ret = np.zeros(shape=[L,L,1])
  coord_x, coord_y = coord
  coord_x_idx = np.argmax(coord_x)
  coord_y_idx = np.argmax(coord_y)
  ret[coord_x_idx][coord_y_idx][0] = color

  draw(ret, name)
  

def draw(m, name, extra=None):
  FIG.clf()

  matrix = m
  orig_shape = np.shape(matrix)
  # lose the channel shape in the end of orig_shape
  new_shape = orig_shape[:-1] 
  matrix = np.reshape(matrix, new_shape)
  ax = FIG.add_subplot(1,1,1)
  ax.set_aspect('equal')
  plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.gray)
  # plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.ocean)
  plt.colorbar()

  if extra != None:
    greens, reds = extra
    grn_x, grn_y, = greens
    red_x, red_y = reds
    plt.scatter(x=grn_x, y=grn_y, c='g', s=40)
    plt.scatter(x=red_x, y=red_y, c='r', s=40)
#  # put a blue dot at (10, 20)
#  plt.scatter([10], [20])
#  # put a red dot, size 40, at 2 locations:
#  plt.scatter(x=[3, 4], y=[5, 6], c='r', s=40)
#  # plt.plot()

  plt.savefig(name)

def draw_orig(img, name):
  L1, L2 = img.shape
  ret = np.reshape(img, [L1,L2,1])
  draw(ret, name)

# draws in image space
def draw_allob(img, name, ob_prefix):
  
  ret = np.zeros([L,L,1])
  for ii in range(L):
    for jj in range(L):
      labb = 0.5 if img[ii][jj][0] == img[ii][jj][1] else img[ii][jj][0]
      # labb = img[ii][jj][0]
      # labb = img[ii][jj][0] - img[ii][jj][1]
      ret[ii][jj][0] = labb

  grn_x = []
  grn_y = []
  red_x = []
  red_y = []

  for obob in ob_prefix:
    ob_c, labb = obob
    if labb[0] > labb[1]:
      grn_x.append(ob_c[0])
      grn_y.append(ob_c[1])
    else:
      red_x.append(ob_c[0])
      red_y.append(ob_c[1])

  draw(ret, name, ((grn_y, grn_x), (red_y, red_x)))

def draw_obs(obs, name):
  ret_shape = [L, L, 1]
  ret = np.zeros(shape=ret_shape)
  for ob, lab in obs:
    ii, jj = ob
    labb = 1.0 if lab[0] > lab[1] else -1.0
    # labb = lab[0]
    ret[ii][jj][0] = labb
  draw(ret, name)

def draw_annotate(x_cords, y_cords, anns, name):
  FIG.clf()
  y = x_cords
  z = y_cords
  n = anns
  fig = FIG
  ax = fig.add_subplot(1,1,1)
  ax.set_xlim([0,L])
  ax.set_ylim([0,L])
  ax.set_ylim(ax.get_ylim()[::-1])
  ax.scatter(z, y)

  for i, txt in enumerate(n):
    ax.annotate(txt, (z[i],y[i]))
  fig.savefig(name)

def draw_obs_trace(obs, name):
  x_coords = []
  y_coords = []
  anno = []
  for i, ob in enumerate(obs):
    ob_coord, ob_outcome = ob
    x_coords.append(ob_coord[0])
    y_coords.append(ob_coord[1])
    anno.append("O"+str(i)+str(int(ob_outcome[0])))

  draw_annotate(x_coords, y_coords, anno, name)

def draw_all_preds(all_preds, name):
  ret_shape = [L, L, 1]
  ret = np.zeros(shape=ret_shape)

  for qq, labb in all_preds:
    i, j = qq
    # ret[i][j][0] = 1.0 if labb[0] > labb[1] else 0.0
    # ret[i][j][0] = labb[0]
    ret[i][j][0] = labb[0]
  
  draw(ret, name)
  

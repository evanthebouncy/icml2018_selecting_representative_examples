import numpy as np
import random

from meta_params import *

# L = 64
# S_WIDTHS = [2,3,5]


# null shape
def mk_null():
  def null(x,y):
    return False
  return null

# takes in parameters: center x, center y, width
# returns a predicate function that draws the square
def mk_square(s_x,s_y,w):
  def square(x,y):
    return s_x-w<=x<=s_x+w and s_y-w<=y<=s_y+w
  return square

# takes in parameters: start x, start y, terminal x, terminal y
# returns a predicate function that draws the line
def mk_line(s_x,s_y,t_x,t_y,l_kind):
  x_min, x_max = min(s_x,t_x), max(s_x,t_x)
  y_min, y_max = min(s_y,t_y), max(s_y,t_y)

  def line(x,y):
    # if out of bound then no way
    if x < x_min or x > x_max or y < y_min or y > y_max:
      return False
    top_edge = y == y_min
    right_edge = x == x_max
    left_edge = x == x_min
    bot_edge = y == y_max

    l1 = top_edge or right_edge
    l2 = right_edge or bot_edge
    l3 = bot_edge or left_edge
    l4 = left_edge or top_edge

    na = (s_x == x_min and s_y == y_min) or (s_x == x_max and s_y == y_max)

    if na and l_kind: return l1
    if na and not l_kind: return l3
    if not na and l_kind: return l2
    if not na and not l_kind: return l4

  return line

# takes in parameters: start x, start y, terminal x, terminal y
# returns a predicate function that draws the line
def mk_line_diag(s_x,s_y,t_x,t_y):
  x_min, x_max = min(s_x,t_x), max(s_x,t_x)
  y_min, y_max = min(s_y,t_y), max(s_y,t_y)
  line_diffx = t_x - s_x
  line_diffy = t_y - s_y

  def line(x,y):
    # if out of bound then no way
    if x < x_min or x > x_max or y < y_min or y > y_max:
      return False

    diffx, diffy = x - s_x, y - s_y
    err = abs(diffx * line_diffy - diffy * line_diffx) 

    def left_right_logic():
      up_diffy   = diffy - 1
      down_diffy = diffy + 1

      up_err = abs(diffx * line_diffy - up_diffy * line_diffx) 
      down_err = abs(diffx * line_diffy - down_diffy * line_diffx) 

      return err <= up_err and err <= down_err

    def up_down_logic():
      left_diffx = diffx - 1
      right_diffx = diffx + 1
      left_err = abs(left_diffx * line_diffy - diffy * line_diffx)
      right_err = abs(right_diffx * line_diffy - diffy * line_diffx)

      return err <= left_err and err <= right_err

    return left_right_logic() or up_down_logic()

  return line

# takes in 3 integers a,b,c
# returns a linear transform on i, j
# output a single value
def mk_xform(a,b,c):
  def xform(i,j):
    return a * i + b * j + c
  return xform

# takes in a set of parameters
# returns a function that
# when takes in i, j as arguments
# produce a coordinate offset transform
# @ arguments: oxa, oxb, oxc = xform args for the ox prameter
# -----------: oya, oyb, oyc = xform args for the oy prameter
def mk_coord_xform(oxa,oxb,oxc, oya,oyb,oyc):
  ox_xform = mk_xform(oxa,oxb,oxc)
  oy_xform = mk_xform(oya,oyb,oyc)
  def mk_xform_coord(i,j): 
    xformed_x = ox_xform(i,j)
    xformed_y = oy_xform(i,j)
    return xformed_x, xformed_y

  return mk_xform_coord

# given an coord x and y 
# produce a square with offset and width
def mk_sq_from_coord(coord_x, coord_y, offset_x, offset_y, w):
  return mk_square(coord_x + offset_x,
                   coord_y + offset_y,w)


# given coord x and y
# produce a line with start offset, end offset
def mk_line_from_coord(coord_x, coord_y, i, j,
                       start_x, start_y,
                       end_x, end_y,
                       supress_i, supress_j, line_kind):
  if supress_i and i == 0:
    return mk_null()
  if supress_j and j == 0:
    return mk_null()
  return mk_line(coord_x + start_x, coord_y + start_y,
                 coord_x + end_x, coord_y + end_y, line_kind)

# render shapes onto a L by L canvas
# render goes from x,y space to id1,id2 space by flipping
def render(shapes):
  canvas = np.zeros([L,L])

  for y in range(L):
    for x in range(L):
      for s in shapes:
        if s(x,y):
          canvas[y][x] = 1

  return canvas

# --------------------------- generators -------------------------- #

def sample_coord_xform_params():
  base_choice = range(TR_LOW_BND, TR_HIGH_BND+1)
  x_a = random.choice(base_choice)
  x_b = random.choice(base_choice)
  x_c = random.choice(base_choice)
  y_a = random.choice(base_choice)
  y_b = random.choice(base_choice)
  y_c = random.choice(base_choice)
  return [x_a, x_b, x_c, y_a, y_b, y_c]

def sample_square_params():
  base_choice = range(SQ_LOW_BND, SQ_HIGH_BND+1)
  offset_x = random.choice(base_choice)
  offset_y = random.choice(base_choice)
  w = random.choice(S_WIDTHS)
  return [offset_x, offset_y, w]

def sample_line_params():
  def sample_bool():
    return [ random.choice([True, False]) ]

  base_choice = range(SQ_LOW_BND, SQ_HIGH_BND+1)
  s_x = random.choice(base_choice)
  s_y = random.choice(base_choice)
  t_x = random.choice(base_choice)
  t_y = random.choice(base_choice)
  return [s_x, s_y, t_x, t_y]\
          + sample_bool()\
          + sample_bool()\
          + sample_bool()


def sample_iter():
  return random.choice(range(1, N_ITERS+1))

def square_no_overlap(squares):
  for i in range(L):
    for j in range(L):
      preds = [1 if s(i,j) else 0 for s in squares]
      if sum(preds) > 1:
        return False
  return True 
  
def sample_params():
  num_i_iter = sample_iter()
  num_j_iter = sample_iter()

  coord_params  = sample_coord_xform_params()
  square_params = [sample_square_params()\
                   for _ in range(N_SQUARES)]
  line_params   = [sample_line_params()\
                   for _ in range(N_LINES)]

  params = {}
  params['iter_i'] = num_i_iter
  params['iter_j'] = num_j_iter
  params['transforms'] = coord_params
  params['squares'] = square_params
  params["lines"] = line_params
  return params
  
def mk_scene(params):

  iter_i = params['iter_i']
  iter_j = params['iter_j']
  transforms = params['transforms']
  square_params = params['squares']
  line_params = params["lines"]

  coord_xform = mk_coord_xform(*transforms)

  squares = []
  lines = []
  for i in range(iter_i):
    for j in range(iter_j):
      coord_x, coord_y = coord_xform(i,j)
      for s_params in square_params:
        square = mk_sq_from_coord(coord_x, coord_y, *s_params)
        squares.append(square)
      for l_params in line_params:
        line = mk_line_from_coord(coord_x, coord_y, i, j, *l_params)
        lines.append(line)

  return squares, lines

if __name__ == "__main__":

  from draw import *
  import time
  import hand_examples
  # params = hand_examples.ex5
  params = sample_params()
  squares_orig,lines_orig = mk_scene(params)
  rendered = render(squares_orig + lines_orig)

  print params
  draw_orig(rendered, "hand_drawings/sampled.png")


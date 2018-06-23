from z3 import *
import numpy as np
from util import *
import time
from data import *
from draw import *
from graphix_lang import * 

def Max(solver,x,y):
  z = FreshInt()
  solver.add(z == z3.If(x < y, y, x))
  return z

def Min(solver,x,y):
  z = FreshInt()
  solver.add(z == z3.If(x > y, y, x))
  return z

class Square(object):

    def __init__(self, center_x, center_y, width, run_at_ij, square_exist):
        self.center_x = center_x
        self.center_y = center_y
        self.width = width
        self.run_at_ij = run_at_ij
        self.square_exist = square_exist

    def inside(self, x,y):
        return And(
            self.center_x-self.width<=x, x<=self.center_x+self.width,
            self.center_y-self.width<=y, y<=self.center_y+self.width,
            self.run_at_ij, self.square_exist,
        )

# make a constraint maker for x,y, but also assert some high-level constraints first
def mk_line(solver, 
            off_x_start, 
            off_y_start, 
            off_x_end, 
            off_y_end, 
            supress_i, 
            supress_j, 
            l_kind,
            i, j, run_at_ij, line_exist):

    x_min, x_max = Min(solver,off_x_start, off_x_end), Max(solver,off_x_start, off_x_end)
    y_min, y_max = Min(solver,off_y_start, off_y_end), Max(solver,off_y_start, off_y_end)
    activate_i = FreshBool()
    activate_j = FreshBool()
    solver.add(activate_i == Or(i > 0, Not(supress_i)))
    solver.add(activate_j == Or(j > 0, Not(supress_j)))

    def inside(x,y):
        na = Or(And(off_x_start == x_min , off_y_start == y_min) , 
                And(off_x_start == x_max , off_y_start == y_max))

        constrain_in = And(x >= x_min, x <= x_max, y >= y_min, y <= y_max)
  
        top_edge = y == y_min
        right_edge = x == x_max
        left_edge = x == x_min
        bot_edge = y == y_max

        l1 = Or(top_edge , right_edge) # looks like 7
        l2 = Or(right_edge , bot_edge)
        l3 = Or(bot_edge , left_edge)
        l4 = Or(left_edge , top_edge)

        imp1 = Implies(And(na,l_kind) , l1)
        imp2 = Implies(And(na,Not(l_kind)) , l3)
        imp3 = Implies(And(Not(na),l_kind) , l2)
        imp4 = Implies(And(Not(na),Not(l_kind)) , l4)

        return And(
           run_at_ij, line_exist, 
           activate_i, activate_j,
           constrain_in,
           And(imp1,imp2,imp3,imp4),
           # Or(top_edge, right_edge, left_edge, bot_edge)
           )

    return inside
    

def transform(i,j,a,b,c):
    return i*a + j*b + c

# the solver for taking in constraints and make them into squares
class DrawingSolver:

    def __init__(self):
        self.mk_solver()
        self.seen = set()

    def mk_solver(self):
        self.s = Solver()

        # overall bound on the size of the program
        self.iter_i = Int('iter_bnd_i')
        self.iter_j = Int('iter_bnd_j')
        self.n_squares = Int('n_squares')
        self.n_lines = Int('n_lines')
        self.program_size = Int('program_size')
        # constraints on iterations and n squares and program size
        self.s.add(self.program_size == self.iter_i + self.iter_j + self.n_squares + self.n_lines)
        self.s.add(self.iter_i <= ITER_I_BND)
        self.s.add(self.iter_j <= ITER_J_BND)
        self.s.add(self.iter_i >= 0)
        self.s.add(self.iter_j >= 0)
        self.s.add(0 <= self.n_squares)
        self.s.add(self.n_squares <= N_SQUARES)
        self.s.add(0 <= self.n_lines)
        self.s.add(self.n_lines <= N_LINES)

        # coordinate transformations
        self.transforms = [Int('xform'+c) for c in ['a', 'b', 'c', 'd', 'e', 'f']]
        self.x_transforms = self.transforms[:3]
        self.y_transforms = self.transforms[3:]
        # set range limit on these transforms
        for tr_par in self.transforms:
          self.s.add(tr_par <= TR_HIGH_BND)
          self.s.add(tr_par >= TR_LOW_BND)

        # parameters for the square
        self.c_x, self.c_y, self.w = [],[],[] 
        # the list of square constraints being created
        self.sq_constraints = []

        # parameters for the lines
        self.l_sx, self.l_sy, self.l_tx, self.l_ty, self.sup_i, self.sup_j, self.l_kind = [],[],[],[],[],[],[]
        self.line_constraints = []


        # make some square parameters
        for sq_num in xrange(N_SQUARES):
            sq_offset_x = Int('sq_offset_x_%d' % sq_num)
            sq_offset_y = Int('sq_offset_y_%d' % sq_num)
            s_width = Int('sq_w_%d' % sq_num)

            self.c_x.append(sq_offset_x)
            self.c_y.append(sq_offset_y)
            self.w.append(s_width)

            # constrain offset 
            self.s.add(And([sq_offset_x >= SQ_LOW_BND, sq_offset_x <= SQ_HIGH_BND]))
            self.s.add(And([sq_offset_y >= SQ_LOW_BND, sq_offset_y <= SQ_HIGH_BND]))
            # constrain width
            self.s.add(Or([s_width == w for w in S_WIDTHS]))

        # make some line parameters
        for line_num in xrange(N_LINES):
            l_off_sx = Int('line_off_sx_%d' % line_num)
            l_off_sy = Int('line_off_sy_%d' % line_num)
            l_off_tx = Int('line_off_tx_%d' % line_num)
            l_off_ty = Int('line_off_ty_%d' % line_num)

            sup_i = Bool('line_supress_i_%d' % line_num)
            sup_j = Bool('line_supress_j_%d' % line_num)
            l_kind = Bool('l_kind_%d' % line_num)

            # constrain offset 
            self.s.add(And([l_off_sx >= SQ_LOW_BND, l_off_sx <= SQ_HIGH_BND]))
            self.s.add(And([l_off_sy >= SQ_LOW_BND, l_off_sy <= SQ_HIGH_BND]))
            self.s.add(And([l_off_tx >= SQ_LOW_BND, l_off_tx <= SQ_HIGH_BND]))
            self.s.add(And([l_off_ty >= SQ_LOW_BND, l_off_ty <= SQ_HIGH_BND]))

            self.s.add(l_off_sy <= l_off_ty)

            self.l_sx.append(l_off_sx)
            self.l_sy.append(l_off_sy)
            self.l_tx.append(l_off_tx)
            self.l_ty.append(l_off_ty)
            self.sup_i.append(sup_i)
            self.sup_j.append(sup_j)
            self.l_kind.append(l_kind)

        for i in xrange(ITER_I_BND):
            for j in xrange(ITER_J_BND):
                transform_x = Int('tx_{}_{}'.format(i,j))
                transform_y = Int('ty_{}_{}'.format(i,j))
                self.s.add(transform_x == transform(i,j,*self.x_transforms))
                self.s.add(transform_y == transform(i,j,*self.y_transforms))

                # check if a particular i, j iteration is being executed
                run_at_ij = Bool('run_at_{}_{}'.format(i,j))
                self.s.add(run_at_ij == And([i < self.iter_i, j < self.iter_j]))
                
                # make some constraints for the squares
                for sq_num in xrange(N_SQUARES):
                    # check if this square should exist
                    square_exist = Bool('square_exist_{}'.format(sq_num))
                    self.s.add(square_exist == (sq_num < self.n_squares))
                    self.sq_constraints.append(Square(self.c_x[sq_num]+transform_x, self.c_y[sq_num]+transform_y, self.w[sq_num], 
                                               run_at_ij, square_exist).inside)

                # make some constraints for lines
                for line_num in xrange(N_LINES):
                    # check if this line should exist
                    line_exist = Bool('line_exist_{}'.format(line_num))
                    self.s.add(line_exist == (line_num < self.n_lines))
                    self.line_constraints.append(mk_line(self.s,
                                                         self.l_sx[line_num]+transform_x,
                                                         self.l_sy[line_num]+transform_y,
                                                         self.l_tx[line_num]+transform_x,
                                                         self.l_ty[line_num]+transform_y,
                                                         self.sup_i[line_num],
                                                         self.sup_j[line_num],
                                                         self.l_kind[line_num],
                                                         i,j,run_at_ij,line_exist ) )

    def add_hints(self, hints):
        if 'iter_i' in hints:
          self.s.add(self.iter_i == hints['iter_i'])
        
        if 'iter_j' in hints:
          self.s.add(self.iter_j == hints['iter_j'])

        if 'transforms' in hints:
          zippy = zip(hints['transforms'], self.transforms)
          for xx in zippy:
            self.s.add(xx[1] == xx[0])

    def solve(self, program_size_bnd, constraints, hints={}):
        probes = []

        # print "adding hints . . ."
        self.add_hints(hints)

        start_time = time.time()

        # print "adding {} constraints . . . ".format(len(constraints))
        self.s.add(self.program_size <= P_SIZE)
        # self.s.add(self.program_size <= program_size_bnd)
        for x_y, val in constraints:
          if (x_y, val) not in self.seen:
            self.seen.add((x_y, val)) 
            all_squares_occupy = Or([sq_const(*x_y) for sq_const in self.sq_constraints])
            all_lines_occupy = Or([line_const(*x_y) for line_const in self.line_constraints])
            all_any_occupy = Or(all_squares_occupy, all_lines_occupy)
            self.s.add(all_any_occupy == val)

        model_building_time = time.time() - start_time
        # print "finished adding constraints, solving . . ."

        if self.s.check() == sat:
            model = self.s.model()
            
            solving_time = time.time() - start_time - model_building_time
            ret = {}
            # get the loop iteration information and bounds
            ret['iter_i'] = model[self.iter_i].as_long()
            ret['iter_j'] = model[self.iter_j].as_long()
            ret['n_squares'] = model[self.n_squares].as_long()
            ret['n_lines'] = model[self.n_lines].as_long()
            ret['program_size'] = model[self.program_size].as_long()
            # get the transform information
            ret['transforms'] = [model[xform_param].as_long() for xform_param in self.transforms]
            squares = [[] for _ in range(ret['n_squares'])]
            for square_id in range(ret['n_squares']):
              squares[square_id].append(model[self.c_x[square_id]].as_long())
              squares[square_id].append(model[self.c_y[square_id]].as_long())
              squares[square_id].append(model[self.w[square_id]].as_long())
            ret['squares'] = squares

            lines = [[] for _ in range(ret['n_lines'])]
            for l_id in range(ret['n_lines']):
              lines[l_id].append(model[self.l_sx[l_id]].as_long())
              lines[l_id].append(model[self.l_sy[l_id]].as_long())
              lines[l_id].append(model[self.l_tx[l_id]].as_long())
              lines[l_id].append(model[self.l_ty[l_id]].as_long())
              lines[l_id].append(model[self.sup_i[l_id]].__bool__())
              lines[l_id].append(model[self.sup_j[l_id]].__bool__())
              try:
                lines[l_id].append(model[self.l_kind[l_id]].__bool__())
              except:
                lines[l_id].append(False)
            ret['lines'] = lines

            ret['building_time'] = model_building_time
            ret['solving_time'] = solving_time

            return ret
        else:
            return "UNSAT"

    def solve_grid(self, program_size_bnd, grid):
        print "solving grid "
        (M,N) = grid.shape

        constraints = []
        # the outer loop is actually y and the inner is x
        for y in xrange(M):
            for x in xrange(N):
                value = True if grid[y][x] else False
                constraints.append(((x,y),value))
        return self.solve(program_size_bnd, constraints)

# check returns in index space
def check(params, true_render, i):
  squares, lines = mk_scene(params)
  rendered = render(squares + lines)

  grid_constraints = img_2_labels(rendered)
  # draw_allob(grid_constraints, "hand_drawings/cur_ceigs.png",[])
  # draw_allob(grid_constraints, "hand_drawings/recovered_{}.png".format(i), [])

  diff = rendered - true_render

  # draw_orig(diff, "hand_drawings/diff_{}.png".format(i))

  diff_idx1, diff_idx2 = np.where(diff != 0)

  if len(diff_idx1) == 0:
    return None

  else:
    return zip(diff_idx1, diff_idx2)

def CEGIS(constraints, rendered_squares, rendered_lines, start_constraints = [], hints=[]):

  synth_solver = DrawingSolver()
  sub_constraints = constraints[:1] + start_constraints

  i = 0
  
  while True:
    i += 1
    print sub_constraints
    paras = synth_solver.solve(15, sub_constraints, hints)
    # print "paras"
    # print paras
    ces = check(paras, rendered_squares, rendered_lines, i)
    if ces == None:
      return paras
    else:
      id1,id2 = random.choice(ces)
      square_val = bool(rendered_squares[id1][id2] == 1)
      line_val =   bool(rendered_lines[id1][id2] == 1)
      sub_constraints +=    [((int(id2), int(id1)), 'square', square_val),
                             ((int(id2), int(id1)), 'line', line_val)]
  
   

if __name__ == '__main__':
  # for this simple picture overwrite the width constrain
  S_WIDTHS = [0,1]
  N_SQUARES = 1
  N_LINES = 2
  grid = np.array([
          [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0],
          [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0],
          [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0],
          [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0],
          [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0],
          [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0],
          [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          ])

  grid = np.array([
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0],
          [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
          [0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0],
          [0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0],
          [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
          [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
          [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
          [0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          ])

  grid = np.array([
          [0, 0, 0, 0, 0, 0],
          [0, 1, 1, 1, 0, 0],
          [0, 1, 0, 0, 0, 0],
          [0, 1, 0, 0, 0, 0],
          [0, 1, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0],
          ])

  solver = DrawingSolver()
  print solver.solve_grid(6, grid)

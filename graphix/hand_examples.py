ex_box_line1 = {
  "n_squares" : 3,
  "iter_i" : 3,
  "iter_j" : 3,
  "transforms" : [25, 4, 10, 0, 25, 5],
  "squares" : [[0,0,5], [5,14,3], [-8,12,2]],
  "lines" : [
             [0,0,5,12,False,False, True],
             [0,0,-8,12,False,False, True],
             [0,0,-20,0,True,False, True],
             [-10,12,-20,-8,True,True, True],
            ],
}

ex_box1 = {
  "n_squares" : 3,
  "iter_i" : 3,
  "iter_j" : 3,
  "transforms" : [25, 4, 10, 0, 25, 5],
  "squares" : [[0,0,5], [5,12,3], [-8,12,2]],
  "lines" : []
}



ex_line1 = {
  "n_squares" : 3,
  "iter_i" : 3,
  "iter_j" : 3,
  "transforms" : [25, 4, 10, 0, 25, 5],
  "squares" : [],
  "lines" : [
             [0,0,5,12,False,False, True],
             [0,0,-8,12,False,False, True],
             [0,0,-20,0,True,False, True],
             [-10,12,-20,-8,True,True, True],
            ],
}

ex_scale = {
  "n_squares" : 2,
  "iter_i" : 3,
  "iter_j" : 3,
  "transforms" : [10, 2, 3, 2, 10, 3],
  "squares" : [
                [0,0,2], 
                [5,5,2], 
              ],
  "lines" : [
            ],
}

ex_grid = {
  "n_squares" : 1,
  "iter_i" : 3,
  "iter_j" : 3,
  "transforms" : [10, 0, 5, 0, 10, 0],
  "squares" : [
                [0,5,2], 
              ],
  "lines" : [
               [0,2,0,-7, False,True, True],
               [0,5,-7,5,True,False, True],
            ],
}

ex4 = {
  "n_squares" : 2,
  "iter_i" : 2,
  "iter_j" : 2,
  "transforms" : [10, 2, 3, 2, 10, 3],
  "squares" : [
                [0,0,2], 
                [9,9,2], 
              ],
  "lines" : [
              [9,0,0,9, False,False, True],
              [2,-9,-9,2,False,False, True],
            ],
}

ex5 = {
  "iter_i" : 2,
  "iter_j" : 2,
  "transforms" : [10, 8, 5, 9, 3, 8],
  "squares" : [
                [0,-6,2], 
               [0,8,2], 
              ],
  "lines" : [
              [-6,-6,6,6, False, False, False],
#              [2,-9,-9,2,False,False, True],
            ],

}

draw_params = {
  "I" : 3,
  "J" : 3,
  "affine_xform" : [10, 0, 5, 0, 10, 0],
  "squares" : [
                [0,5], 
              ],
  "lines" : [
               [0,2,0,-7,False,True],
               [0,5,-7,5,True,False],
            ],
}

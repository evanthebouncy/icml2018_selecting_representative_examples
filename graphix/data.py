from graphix_lang import *
from util import *

N_BATCH = 20

def rand_data(epi):

  partial_obss = []
  full_obss = []

  for bb in range(N_BATCH):
    # generate a hidden variable X
    # get a single thing out
    params = sample_params()
    s_orig, l_orig = mk_scene(params)
    rendered = render(s_orig + l_orig)
    qry = mk_query(rendered)
    partial_obs = np.zeros([L,L,2])
    full_obs = np.zeros([L,L,2])
    for i in range(L):
      for j in range(L):
        full_obs[i][j] = qry((i,j))
        if np.random.random() < epi:
          partial_obs[i][j] = qry((i,j))
    partial_obss.append(partial_obs)
    full_obss.append(full_obs)

  return  np.array(partial_obss),\
          np.array(full_obss)

# while True:
#   rand_data(0.1, 2)
#   print "HAHA!"

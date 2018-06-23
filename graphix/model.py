import tensorflow as tf
import numpy as np
from data import *
from util import *
from draw import *

n_hidden = 20

def weight_variable(shape, name=None):
  initial = tf.truncated_normal(shape, stddev=0.1)
  if name:
    return tf.Variable(initial, name)
  else:
    return tf.Variable(initial)

def bias_variable(shape, name=None):
  initial = tf.constant(0.1, shape=shape)
  if name:
    return tf.Variable(initial, name)
  else:
    return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def embed_cnn_layer(input_var):

  W_conv = weight_variable([7, 7, 2, n_hidden])
  b_conv = bias_variable([n_hidden])

  xxx_R17 = conv2d(input_var, W_conv)
  cur = tf.nn.relu(xxx_R17 + b_conv)

  print cur.get_shape() 
  return cur


# transform constraint (x, y)->val into image (id2 id1)->[1.0,0.0] (reversing order)
def optional_xform(obs):
  if obs == [] or len(obs[0]) == 2: return obs
  id1_id2_to_value = {}
  for ob in obs:
    x_y, stype, val = ob
    id1,id2 = rev(*x_y)
    if (id1,id2) not in  id1_id2_to_value:
      id1_id2_to_value[(id1,id2)] = False
    id1_id2_to_value[(id1,id2)] = id1_id2_to_value[(id1,id2)] or val
  return [(id1_id2, [1.0, 0.0] if id1_id2_to_value[id1_id2] else [0.0, 1.0])\
           for id1_id2 in id1_id2_to_value]

class Implynet:

  def gen_feed_dict(self, partial_obs, full_obs):
    ret = dict()
    ret[self.partial_obs] = partial_obs
    ret[self.full_obs] = full_obs
    return ret

  # load the model and give back a session
  def load_model(self, saved_loc):
    sess = self.sess
    self.saver.restore(sess, saved_loc)
    print("Model restored.")

  # make the model
  def __init__(self, sess):
    self.name = 'imply'
    with tf.variable_scope(self.name) as scope:
      # set up placeholders
      self.partial_obs = tf.placeholder(tf.float32, [N_BATCH, L , L , 2], name="partial_obs")
      self.full_obs = tf.placeholder(tf.float32, [N_BATCH, L , L , 2], name="full_obs")

      # embed the input
      embedded = embed_cnn_layer(self.partial_obs)

      unstack_once = tf.unstack(embedded, axis=1)
      print len(unstack_once)
      print unstack_once[0]
      unstack_twice = [tf.unstack(x,axis=1) for x in unstack_once]
      print unstack_twice[L-1][L-1]
    
      # do the prediction on top of the embedding
      W_pred = weight_variable([n_hidden, 2])
      b_pred = bias_variable([2])
      e2 = 1e-10

      print "building prediction for each slice "
      self.query_preds = []
      for iii in range(L):
        for jjj in range(L):
          same_line_print("predictor "+str(iii)+" "+str(jjj))
          slicey = unstack_twice[iii][jjj]
          query_pred = tf.nn.softmax(tf.matmul(slicey, W_pred) + b_pred)+e2
          self.query_preds.append(query_pred)
      # print "query_preds shape ", show_dim(self.query_preds)

      # doing some reshape of the input tensor
      full_obs_trans = tf.transpose(self.full_obs, perm=[1,2,0,3])
      print full_obs_trans.get_shape()
      full_obs_split = tf.reshape(full_obs_trans, [L*L, N_BATCH, 2])
      full_obs_split = tf.unstack(full_obs_split)
      # print show_dim(full_obs_split) 
      print "splitted observation "

      self.query_pred_costs = []
      for idx in range(L * L):
        same_line_print("doing cost " + str(idx))
        blah = -tf.reduce_sum(full_obs_split[idx] * tf.log(self.query_preds[idx]))
        self.query_pred_costs.append(blah)
        
      # print "costs shapes ", show_dim(self.query_pred_costs)
      self.cost_query_pred = sum(self.query_pred_costs)

      # ------------------------------------------------------------------------ training steps
      optimizer = tf.train.AdamOptimizer(0.001)


      print "computing gradient "
      pred_gvs = optimizer.compute_gradients(self.cost_query_pred)
      capped_pred_gvs = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in pred_gvs]
      #train_pred = optimizer.minimize(cost_pred, var_list = VAR_pred)
      self.train_query_pred = optimizer.apply_gradients(capped_pred_gvs)

      # train_query_pred = optimizer.minimize(cost_query_pred, var_list = VAR_pred)
      # Before starting, initialize the variables.  We will 'run' this first.
      self.init = tf.global_variables_initializer()
      self.saver = tf.train.Saver()
      self.sess = sess
      print "computation graph completely built "

  def initialize(self):
    self.sess.run(self.init)

  # save the model
  def save(self):
    model_loc = "./models/" + self.name+".ckpt"
    sess = self.sess
    save_path = self.saver.save(sess, model_loc)
    print("Model saved in file: %s" % save_path)

  # train on a particular data batch
  def train(self, data_batch):
    sess = self.sess

    partial_obs, full_obs = data_batch
    feed_dic = self.gen_feed_dict(partial_obs, full_obs)

    cost_query_pred_pre = sess.run([self.cost_query_pred], feed_dict=feed_dic)[0]
    sess.run([self.train_query_pred], feed_dict=feed_dic)
    cost_query_pred_post = sess.run([self.cost_query_pred], feed_dict=feed_dic)[0]
    print "train query pred ", cost_query_pred_pre, " ",\
      cost_query_pred_post, " ", True if cost_query_pred_post < cost_query_pred_pre else False

  # =========== HELPERS =============

  # a placeholder to feed in a single observation
  def get_feed_dic_obs(self, obs):
    num_obs = len(obs)
    _obs = np.zeros([L,L,2])

    for ob_idx in range(num_obs):
      cord, lab = obs[ob_idx]
      xx, yy = cord
      _obs[xx][yy] = lab

    import time
    # draw_allob(_obs, "hand_drawings/cur_obs{}.png".format(time.time()),[])
    
    obss = np.array([_obs for i in range(N_BATCH)])

    feed_dic = dict()
    feed_dic[self.partial_obs] = obss
    return feed_dic

  def get_all_preds(self, obs):
    sess = self.sess
    dick = self.get_feed_dic_obs(obs)
    predzz = sess.run(self.query_preds, dick)
    predzz0 = np.array([x[0] for x in predzz])
    predzz0 = np.reshape(predzz0, [L,L,2])
    return predzz0

  def get_most_confuse(self, sess, obs):
    obs_qry = [_[0] for _ in obs]
    all_preds = self.get_all_preds(obs)
    
    all_pred_at_key1 = []
    for i in range(L):
      for j in range(L):
        qry = i, j
        value = all_preds[i][j]
        if qry not in obs_qry:
          all_pred_at_key1.append((qry, value))
    most_confs = [(abs(x[1][0] - x[1][1]), x[0]) for x in all_pred_at_key1]
    most_conf = min(most_confs)
    return most_conf[1]

  # this should work over index space, i.e. id1,id2 directly onto image. flip all constraints
  # around to this form
  def get_sorted_unlikely(self, sub_constraints, full_image, i=0):
    constraints = img_2_labels(full_image)
    # if this is called from a solver end these can be booleans, convert back to labels
    sub_constraints = optional_xform(sub_constraints)

    all_preds = self.get_all_preds(sub_constraints)
    import time
    # draw_allob(all_preds, "hand_drawings/pred_{}.png".format(time.time()),[])

    constraint_probs = np.sum(all_preds * constraints, axis=2)
    
    for chosen in sub_constraints:
      loc, val = chosen
      constraint_probs[loc[0]][loc[1]] = 1.0

    to_sort = []
    for id1 in range(L):
      for id2 in range(L):
        prob = constraint_probs[id1][id2]
        to_sort.append((prob+1e-5*random.random(), (id1,id2)))
    
    return sorted(to_sort)

  def get_trace(self, r_img, increment=10, confidence=0.9):
    query = mk_query(r_img)
    M,N = r_img.shape
    obs = []

    bound = M*N
    for i in range(bound):
      sorted_qrys = self.get_sorted_unlikely(obs, r_img)
      # min_prob = sorted_qrys[0][0]
      avg_prob = np.mean([x[0] for x in sorted_qrys])
      sorted_qrys = sorted_qrys[:increment]
      chosen_qrys = [xx[1] for xx in sorted_qrys]
      # print i, avg_prob, confidence
      for chosen_qry in chosen_qrys:
        obs.append((chosen_qry, query(chosen_qry)))

      all_preds = self.get_all_preds(obs)
      # draw_allob(all_preds, "drawings/pred{0}.png".format(i), [])

      # constraints = 
      # constraint_probs = np.sum(all_preds * constraints, axis=2)
      # draw_orig(constraint_probs, "drawings/constraint_prob{0}.png".format(i))

      # if min_prob > 0.3:
      #   break
      if avg_prob > confidence:
        break
    return obs



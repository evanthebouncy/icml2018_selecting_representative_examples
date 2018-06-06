import tensorflow as tf
from gen import *
import numpy as np
import random

num_hidden = 1000

class Oracle:

  def make_graph(self):
    self.session = tf.Session()
    print "making graph "
    self.observed_orders = tf.placeholder(tf.float32, [None, L, L, 2])

    observed_flat = tf.reshape(self.observed_orders,\
        [tf.shape(self.observed_orders)[0], L*L*2])

    print observed_flat.get_shape()

    hidden = tf.layers.dense(observed_flat, num_hidden, activation=tf.nn.relu)

    # a brand new string w/o any label, just a string of 0 and 1
    self.query_x = tf.placeholder(tf.float32, [None, L])
    self.query_y = tf.placeholder(tf.float32, [None, L])
    query = tf.concat([self.query_x, self.query_y], 1)
    
    hidden_and_query = tf.concat([hidden, query], 1)
    print "hidden with query ", hidden_and_query.get_shape()
    hidden_pred = tf.layers.dense(hidden_and_query, num_hidden, activation=tf.nn.relu)
    print "pass another dense ", hidden_pred.get_shape()

    self.prediction = tf.layers.dense(hidden_pred, 2)
    self.pred_prob = tf.nn.softmax(self.prediction)
    print "prediction shape ", self.pred_prob.get_shape()

    # the labeled truth
    self.query_TF = tf.placeholder(tf.float32, [None, 2])

    # add a small number so it doesn't blow up (logp or in action selection)
    self.pred_prob = self.pred_prob + 1e-8

    # set up the cost function for training
    self.log_pred_prob = tf.log(self.pred_prob)
    self.objective = tf.reduce_mean(self.log_pred_prob * self.query_TF)

    self.loss = -self.objective

    self.optimizer = tf.train.AdamOptimizer(0.0001)
    self.train = self.optimizer.minimize(self.loss)

    initializer = tf.global_variables_initializer()
    self.session.run(initializer)

    self.saver = tf.train.Saver()

  def __init__(self, name):
    print "hello "
    self.name = name
    self.make_graph()

  def restore_model(self, path):
    self.saver.restore(self.session, path)
    print "model restored  from ", path

  # save the model
  def save(self):
    model_loc = "./models/tmp/" + self.name+".ckpt"
    sess = self.session
    save_path = self.saver.save(sess, model_loc)
    print("Model saved in file: %s" % save_path)

  def learn(self, b_ob, b_q_x, b_q_y, b_tf):
    stuffs = [b_ob, b_q_x, b_q_y, b_tf]
    # for s in stuffs:
    #   print s.shape
    loss_train = self.session.run([self.loss, self.train], 
                                   {
                                     self.observed_orders : b_ob,
                                     self.query_x : b_q_x,
                                     self.query_y : b_q_y,
                                     self.query_TF: b_tf,
                                   }
                                  )
    print " LOSS: ", loss_train[0]

  def seq_learn(self, strs, TFs):
    assert 0, "UROD BLYAT"

  def predict(self, b_ob, b_q_x, b_q_y):
    stuffs = [b_ob, b_q_x, b_q_y]
    # for s in stuffs:
    #   print s.shape
    pred = self.session.run([self.pred_prob], 
                                   {
                                     self.observed_orders : b_ob,
                                     self.query_x : b_q_x,
                                     self.query_y : b_q_y,
                                   }
                                  )[0]
    return pred

  def get_most_unlikely(self, observed, unobserved):
    b_ob, b_q_x, b_q_y, b_tf = [],[],[],[]

    for u_o in unobserved:
      ob = observed_to_np(observed)
      qx, qy, qtf = query_to_np(u_o)
      b_ob.append(ob)
      b_q_x.append(qx)
      b_q_y.append(qy)
      b_tf.append(qtf)

    preds = self.predict(b_ob, b_q_x, b_q_y)
    all_probs = [np.dot(x[0],x[1]) + random.random()*1e-5 for x in zip(preds, b_tf)]
    abc = zip(all_probs, unobserved)
    return sorted(abc)

  def get_until_confident(self, all_observations, confidence=0.9):
    # be aggressive with increment
    increment = 1
    observed =   []
    unobserved = all_observations

    unlikely = self.get_most_unlikely(observed, unobserved)

    # a how many points are uncertain
    def uncertain(unlikelies):
      return len([x[0] for x in unlikelies if x[0] < confidence])

    conf = []

    while uncertain(unlikely) > int(0.05 * len(all_observations)):
      print len(observed), len(unobserved), uncertain(unlikely)
      observed +=  [x[1] for x in unlikely[:increment]]
      unobserved = [x[1] for x in unlikely[increment:]]
      if len(unobserved) == 0: break
      unlikely = self.get_most_unlikely(observed, unobserved)
      conf.append(uncertain(unlikely))

    print conf
    print [x[0] for x in unlikely]
    return observed
    

if __name__ == "__main__":
  from gen import *
  oracle = Oracle("oracle")

  for i in range(1000000):
    if i % 100 == 1:
      print i
      oracle.save()

    oracle.learn(*gen_batch_data(40))



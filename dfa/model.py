import tensorflow as tf
from gen import *
import numpy as np
import random

num_hidden = 1000

class Oracle:

  def make_graph(self):
    self.session = tf.Session()
    print "making graph "
    self.observed_front_strs = tf.placeholder(tf.float32, [None, N_CLOSE, L, N_CHAR])
    self.observed_front_TFs  = tf.placeholder(tf.float32, [None, N_CLOSE, 2])
    self.observed_back_strs = tf.placeholder(tf.float32, [None, N_CLOSE, L, N_CHAR])
    self.observed_back_TFs  = tf.placeholder(tf.float32, [None, N_CLOSE, 2])

    flatten_front_strs = tf.reshape(self.observed_front_strs,\
        [tf.shape(self.observed_front_strs)[0], N_CLOSE*L*N_CHAR])
    flatten_front_TFs  = tf.reshape(self.observed_front_TFs,\
        [tf.shape(self.observed_front_TFs)[0], N_CLOSE*2])
    flatten_front = tf.concat([flatten_front_strs, flatten_front_TFs], 1)

    print "flat front str ", flatten_front_strs.get_shape()
    print "flat front tf ", flatten_front_TFs.get_shape()
    print "flat front togeter ", flatten_front.get_shape()

    flatten_back_strs = tf.reshape(self.observed_back_strs,\
        [tf.shape(self.observed_back_strs)[0], N_CLOSE*L*N_CHAR])
    flatten_back_TFs  = tf.reshape(self.observed_back_TFs,\
        [tf.shape(self.observed_back_TFs)[0], N_CLOSE*2])
    flatten_back = tf.concat([flatten_back_strs, flatten_back_TFs], 1)

    print "flat back str ", flatten_back_strs.get_shape()
    print "flat back tf ", flatten_back_TFs.get_shape()
    print "flat back togeter ", flatten_back.get_shape()

    hidden_front = tf.layers.dense(flatten_front, num_hidden, activation=tf.nn.relu)
    hidden_back = tf.layers.dense(flatten_back, num_hidden, activation=tf.nn.relu)

    hidden_together = tf.concat([hidden_front, hidden_back], 1)
    print "front and back hidden concat ", hidden_together.get_shape()

    # another hidden state maybe it'll work better
    hidden_state = tf.layers.dense(hidden_together, num_hidden, activation=tf.nn.relu)
    print "hidden state encoding all input info", hidden_state.get_shape()
    
    # a brand new string w/o any label, just a string of 0 and 1
    self.to_decide_strs = tf.placeholder(tf.float32, [None, L, N_CHAR])
    to_decide_flatten = tf.reshape(self.to_decide_strs,\
        [tf.shape(self.to_decide_strs)[0], L * N_CHAR])
    
    hidden_and_to_decide = tf.concat([hidden_state, to_decide_flatten], 1)
    print "hidden with new str input ", hidden_and_to_decide.get_shape()
    hidden_pred = tf.layers.dense(hidden_and_to_decide, num_hidden, activation=tf.nn.relu)
    print "pass another dense ", hidden_pred.get_shape()

    self.prediction = tf.layers.dense(hidden_pred, 2)
    self.pred_prob = tf.nn.softmax(self.prediction)
    print "prediction shape ", self.pred_prob.get_shape()

    # the labeled truth
    self.to_decide_TFs = tf.placeholder(tf.float32, [None, 2])

    # add a small number so it doesn't blow up (logp or in action selection)
    self.pred_prob = self.pred_prob + 1e-8

    # set up the cost function for training
    self.log_pred_prob = tf.log(self.pred_prob)
    self.objective = tf.reduce_mean(self.log_pred_prob * self.to_decide_TFs)

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
    model_loc = "./models/tmp" + self.name+".ckpt"
    sess = self.session
    save_path = self.saver.save(sess, model_loc)
    print("Model saved in file: %s" % save_path)

  def learn(self, b_f_s, b_f_tf, b_b_s, b_b_tf, b_dec_s, b_dec_tf):
    stuffs = [b_f_s, b_f_tf, b_b_s, b_b_tf, b_dec_s, b_dec_tf]
    # for s in stuffs:
    #   print s.shape
    loss_train = self.session.run([self.loss, self.train], 
                                   {
                                    self.observed_front_strs: b_f_s,
                                    self.observed_front_TFs:  b_f_tf,
                                    self.observed_back_strs:  b_b_s,
                                    self.observed_back_TFs:   b_b_tf,
                                    self.to_decide_strs:      b_dec_s,
                                    self.to_decide_TFs:       b_dec_tf,
                                   }
                                  )
    print " LOSS: ", loss_train[0]

  def seq_learn(self, strs, TFs):
    assert 0, "UROD BLYAT"

  def predict(self, b_f_s, b_f_tf, b_b_s, b_b_tf, b_dec_s):
    stuffs = [b_f_s, b_f_tf, b_b_s, b_b_tf, b_dec_s]
    # for s in stuffs:
    #   print s.shape
    pred = self.session.run([self.pred_prob], 
                                   {
                                    self.observed_front_strs: b_f_s,
                                    self.observed_front_TFs:  b_f_tf,
                                    self.observed_back_strs:  b_b_s,
                                    self.observed_back_TFs:   b_b_tf,
                                    self.to_decide_strs:      b_dec_s,
                                   }
                                  )[0]
    return pred

  def get_most_unlikely(self, observed, unobserved):
    b_f_s, b_f_tf, b_b_s, b_b_tf, b_dec_s, b_dec_tf = [],[],[],[],[],[]
    for u_o in unobserved:
      front, back = get_close(observed, u_o)
      front, back = [x[1] for x in front], [x[1] for x in back]
      f_s, f_tf, b_s, b_tf, dec_s, dec_tf = data_to_numpy(front, back, u_o)

      b_f_s.append(f_s)
      b_f_tf.append(f_tf)
      b_b_s.append(b_s)
      b_b_tf.append(b_tf)
      b_dec_s.append(dec_s)
      b_dec_tf.append(dec_tf)

    preds = self.predict(b_f_s, b_f_tf, b_b_s, b_b_tf, b_dec_s)
    all_probs = [np.dot(x[0],x[1]) + random.random()*1e-5 for x in zip(preds, b_dec_tf)]
    abc = zip(all_probs, unobserved)
    return sorted(abc)

  def get_until_confident(self, all_observations, increment=N_CLOSE, confidence=0.9):
    # be aggressive with increment
    increment = int(len(all_observations) / 20)
    observed =   all_observations[:increment]
    unobserved = all_observations[increment:]

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
  ffront, bback, to_decide = gen_train_data()

  for i in range(100000):
    if i % 100 == 1:
      # preds = oracle.predict(train_ob_in, train_ob_out, unob_in)
      # n_correct = 0
      # for xxx in  zip(preds, unob_out):
      #   pp,tt = xxx
      #   if np.argmax(pp) == np.argmax(tt): n_correct += 1
      # print n_correct, len(unob_out)

      # confident_set = oracle.get_until_confident(eee)

      # print "confident subset size ", len(confident_set)
      oracle.save()

    oracle.learn(*gen_batch_data(20))



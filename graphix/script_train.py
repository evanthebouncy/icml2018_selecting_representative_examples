from model import *
from draw import *
import sys


# start the training process some arguments
restart = True

def train_model(impnet, epoch=3000):
  print "training model on randomly generated anticipation tasks"

  if restart:
    impnet.initialize()
  else:
    impnet.load_model(sess, self.name+".ckpt")

  for i in xrange(epoch):
    epi = np.random.random()
    print i, " ", epi
    impnet.train(rand_data(epi))

    if i % 20 == 0:
      partial, full = rand_data(epi)
      predzz = impnet.sess.run(impnet.query_preds, 
                               impnet.gen_feed_dict(partial, full))
      predzz0 = np.array([x[0] for x in predzz])
      # print show_dim(predzz0)
      predzz0 = np.reshape(predzz0, [L,L,2])
      draw_allob(predzz0, "drawings/recovered_ob.png", [])
      draw_allob(full[0], "drawings/orig_ob.png", [])
      draw_allob(partial[0], "drawings/partial_ob.png", [])
      impnet.save()

impnet = Implynet(tf.Session())
train_model(impnet, 20000)


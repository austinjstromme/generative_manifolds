import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.examples.tutorials.mnist import input_data

tfd = tfp.distributions


def make_encoder(data, code_size):
  x = tf.layers.flatten(data)
  x = tf.layers.dense(x, 200, tf.nn.relu)
  x = tf.layers.dense(x, 200, tf.nn.relu)
  loc = tf.layers.dense(x, code_size)
  scale = tf.layers.dense(x, code_size, tf.nn.softplus)
  return tfd.MultivariateNormalDiag(loc, scale)


def make_prior(code_size):
  loc = tf.zeros(code_size)
  scale = tf.ones(code_size)
  return tfd.MultivariateNormalDiag(loc, scale)

def generate_samples(n_l, n_d, num):
  samp = np.random.multivariate_normal(np.zeros(n_l), np.eye(n_l), num)
  res = [np.zeros(n_d) for i in range(0, num)]
  for i in range(0, num):
    samp[i] = samp[i] / np.linalg.norm(samp[i])
    for j in range(0, n_l):
      res[i][j] = samp[i][j]

  return res


def make_decoder(code, data_shape):
  x = code
  x = tf.layers.dense(x, 200, tf.nn.relu)
  x = tf.layers.dense(x, 200, tf.nn.relu)
  #TODO: fix this output
  loc = tf.layers.dense(x, data_shape)
  scale = tf.layers.dense(x, data_shape, tf.nn.softplus)
  return tfd.MultivariateNormalDiag(loc, scale)

def plot_codes(ax, codes, labels):
  ax.scatter(codes[:, 0], codes[:, 1], s=2, c=labels, alpha=0.1)
  ax.set_aspect('equal')
  ax.set_xlim(codes.min() - .1, codes.max() + .1)
  ax.set_ylim(codes.min() - .1, codes.max() + .1)
  ax.tick_params(
      axis='both', which='both', left='off', bottom='off',
      labelleft='off', labelbottom='off')


def plot_samples(ax, samples):
  for index, sample in enumerate(samples):
    ax[index].imshow(sample, cmap='gray')
    ax[index].axis('off')

make_encoder = tf.make_template('encoder', make_encoder)
make_decoder = tf.make_template('decoder', make_decoder)

#params:
ambient_dim = 100
code_size = 2
data_shape = ambient_dim
N = 1000

data = tf.placeholder(tf.float32, [None, ambient_dim])

# Define the model.
prior = make_prior(code_size)
posterior = make_encoder(data, code_size)
code = posterior.sample()

# Define the loss.
likelihood = make_decoder(code, data_shape).log_prob(data)
divergence = tfd.kl_divergence(posterior, prior)
elbo = tf.reduce_mean(likelihood - divergence)
optimize = tf.train.AdamOptimizer(0.001).minimize(-elbo)

samples = make_decoder(prior.sample(10), data_shape).mean()

true_data = generate_samples(code_size, ambient_dim, N)

avg_norm = [0. for i in range(20)]

#fig, ax = plt.subplots(nrows=20, ncols=11, figsize=(10, 20))
with tf.train.MonitoredSession() as sess:
  for epoch in range(20):
    #feed = {data : mnist.test.images.reshape([-1, 28, 28])}
    feed = {data : true_data}
    test_elbo, test_codes, test_samples = sess.run([elbo, code, samples], feed)
    print('Epoch', epoch, 'elbo', test_elbo)
    #ax[epoch, 0].set_ylabel('Epoch {}'.format(epoch))
    #plot_codes(ax[epoch, 0], test_codes, mnist.test.labels)
    #print("samples[0] = " + str(test_samples[0]))
    #print("norm of samples[0] = " + str(np.linalg.norm((test_samples[0]))))
    for j in range(10):
      avg_norm[epoch] += np.linalg.norm(test_samples[j]) / 10.

    #plot_samples(ax[epoch, 1:], test_samples)
    for _ in range(600):
      #feed = {data: mnist.train.next_batch(100)[0].reshape([-1, 28, 28])}
      feed = {data : true_data}
      sess.run(optimize, feed)

print("avg_norm = " + str(avg_norm))
plt.plot(avg_norm)
plt.xlabel('training epoch')
plt.ylabel('average norm')
plt.savefig('avg_norm.png')

#plt.savefig('vae-mnist.png', dpi=300, transparent=True, bbox_inches='tight')
#  for j in range(100):
#    z = np.zeros(code_size)
#    out = make_decoder(prior.sample(10), data_shape).mean()
#    oot = sess.run(out)
#    print("mean of decoder = " + str(oot.mean))
#    print("norm of mean of decoder = " + str(np.linalg.norm(oot.mean)))
#    print()





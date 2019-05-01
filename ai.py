import random
import tensorflow as tf

class AI(object):
    def __init__(self, n_inputs, hidden_neurons=20):
        self.inputs = tf.placeholder(shape=(1,n_inputs), dtype=tf.float32)
        self.W1 = tf.Variable(tf.random_uniform((n_inputs,hidden_neurons), 0, 1))
        self.x1 = tf.matmul(self.inputs, self.W1)
        self.W2 = tf.Variable(tf.random_uniform((hidden_neurons,3), 0, 1))
        self.Q = tf.matmul(self.x1, self.W2)
        self.action = tf.argmax(self.Q, 1)

        self.Q_next = tf.placeholder(shape=self.Q.shape, dtype=tf.float32)
        self.loss = tf.reduce_sum(tf.square(self.Q_next - self.Q))
        self.trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
        self.update_model = self.trainer.minimize(self.loss)

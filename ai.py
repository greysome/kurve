import tensorflow as tf

class AI(object):
    def __init__(self, n_inputs, hidden_neurons, learning_rate):
        self.inputs = tf.placeholder(shape=(1,n_inputs), dtype=tf.float32)
        self.W1 = tf.Variable(tf.random_uniform((n_inputs,hidden_neurons), 0, 1))
        self.x1 = tf.matmul(self.inputs, self.W1)
        self.W2 = tf.Variable(tf.random_uniform((hidden_neurons,3), 0, 1))
        self.Q = tf.reshape(tf.matmul(self.x1, self.W2), (-1,))

        self.W1_ = tf.Variable(self.W1)
        self.x1_ = tf.matmul(self.inputs, self.W1_)
        self.W2_ = tf.Variable(self.W2)
        self.Q_ = tf.reshape(tf.matmul(self.x1_, self.W2_), (-1,))

        self.action = tf.argmax(self.Q, 0)

        self.Q_target = tf.placeholder(shape=self.Q.shape, dtype=tf.float32)
        self.loss = tf.losses.huber_loss(self.Q_target, self.Q)
        self.trainer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.gradient = self.trainer.compute_gradients(self.loss)
        self.update_model = self.trainer.minimize(self.loss)

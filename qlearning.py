import random
import numpy as np
import tensorflow as tf
from ai import AI

class Memory(object):
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = []

    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, n):
        if len(self.buffer) >= n:
            return random.sample(self.buffer, n)
        else:
            return self.buffer

class QLearningSession(object):
    def __init__(self, n_episodes=100,
                 hidden_neurons=20,
                 y=0.9, K=10, learning_rate=100,
                 initial_e=1, final_e=0.1,
                 anneal_frames=10000, observe_frames=1000,
                 update_frames=100, memory_size=5000):
        self.n_inputs = 15
        self.n_episodes = n_episodes

        self.anneal_frames = anneal_frames
        self.observe_frames = observe_frames
        self.update_frames = update_frames
        self.frames = 0

        # future reward discount
        self.y = y
        # exploration-exploitation factor
        self.e = initial_e
        # e value to anneal to over `anneal_frames`
        self.final_e = final_e
        # only train from memory every K iterations
        self.K = K

        # TODO: set learning rate

        self.ai = AI(self.n_inputs,
                     hidden_neurons=hidden_neurons,
                     learning_rate=learning_rate)
        self.memory = Memory(memory_size)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def run(self):
        for i in range(self.n_episodes):
            print(self.sess.run(self.ai.W2)[0,0])
            state, reward, done = self.env_init()
            str_actions = ''
            x = 0

            while True:
                x += 1
                self.frames += 1

                action, _ = self._get_action(state)
                str_actions += str(_)
                new_state, reward, done = self.env_step(action)
                self.memory.add(state, action, reward, new_state, done)

                # update target network
                if self.frames > self.observe_frames and \
                    self.frames % self.update_frames == 0:
                    print('updating Q- model')
                    W1_assign = self.ai.W1_.assign(self.ai.W1)
                    W2_assign = self.ai.W2_.assign(self.ai.W2)
                    self.sess.run((W1_assign, W2_assign))

                # sample memories and train from them
                if self.frames > self.observe_frames and self.frames % self.K == 0:
                    samples = self.memory.sample(3)
                    for s in samples:
                        self._training_step(*s)
                    state = new_state

                self.e -= (self.e-self.final_e) / self.anneal_frames

                if done:
                    print(f'episode {i}')
                    print(f'time alive: {x}')
                    print(f'actions: {str_actions}')
                    print(f'e: {self.e}')
                    print()
                    x = 0
                    break

        self.post_run()

    def env_init(self):
        pass

    def env_step(self):
        pass

    def post_run(self):
        pass

    def _get_action(self, state):
        action = self.sess.run(self.ai.action,
                               feed_dict={self.ai.inputs: state})
        if np.random.rand(1) < self.e:
            return np.random.randint(3), action[0]
        else:
            return action[0], action[0]

    def _get_Q_target(self, state, action, reward, new_state, done):
        Q_target = self.sess.run(self.ai.Q, feed_dict={self.ai.inputs: state})
        if done:
            Q_target[0, action] = reward
        else:
            best_action = self.sess.run(self.ai.action,
                                        feed_dict={self.ai.inputs: new_state})
            Q_ = self.sess.run(self.ai.Q_, feed_dict={self.ai.inputs: new_state})
            Q_target[0, action] = reward + self.y*Q_[0, best_action[0]]
        return Q_target

    def _training_step(self, state, action, reward, new_state, done):
        Q_target = self._get_Q_target(state, action, reward, new_state, done)
        self.sess.run(self.ai.update_model,
                      feed_dict={self.ai.inputs: state,
                                 self.ai.Q_target: Q_target})
        return Q_target[0, action]

import random
import pickle
from sys import argv
import tensorflow as tf
import numpy as np
from tqdm import trange
from engine import Engine
from ai import AI

class TrainSession(object):
    def __init__(self, n_episodes=100, y=0.9, initial_e=1,
                 final_e=0.1, total_frames=10000):
        self.sess = tf.Session()
        self.n_episodes = n_episodes
        self.total_frames = total_frames

        self.ai = AI(12)
        self.engine = Engine(1)


        self.memory = []
        self.max_memory_size = 10000

        # future reward discount
        self.y = 0.9
        # exploration-exploitation factor
        self.e = initial_e
        self.final_e = final_e

    def run(self):
        self.sess.run(tf.global_variables_initializer())

        for i in range(self.n_episodes):
            self.engine.reset()
            state = self.engine.observe(0)
            total_reward = 0

            while True:
                action = self._get_action(state)
                self.engine.step([(0, action)])
                new_state = self.engine.observe(0)

                reward = 0
                if not self._player_is_alive():
                    done = False
                    reward -= 50
                else:
                    done = True
                    reward += 1
                total_reward += reward
                self._add_memory(state, action, reward, new_state, done)

                samples = self._sample_memory(3)
                for s in samples:
                    self._training_step(*s)
                state = new_state

                self.e -= (self.e-self.final_e) / self.total_frames

                if not self._player_is_alive():
                    print(i, 'reward:', total_reward)
                    break

        self._post_run()

    def _player_is_alive(self):
        p = self.engine._find_player(0)
        return p.is_alive

    def _get_action(self, state):
        action = self.sess.run(self.ai.action, feed_dict={self.ai.inputs: state})
        if np.random.rand(1) < self.e:
            return np.random.randint(3)
        else:
            return action[0]

    def _training_step(self, state, action, reward, new_state, done):
        Q_estimate = self.sess.run(self.ai.Q, feed_dict={self.ai.inputs: state})
        if done:
            Q_estimate[0, action] = reward
        else:
            Q1 = self.sess.run(self.ai.Q, feed_dict={self.ai.inputs: new_state})
            Q1_best = np.max(Q1)
            Q_estimate[0, action] = reward + self.y*Q1_best
        _ = self.sess.run(self.ai.update_model,
                          feed_dict={self.ai.inputs: state,
                                     self.ai.Q_next: Q_estimate})

    def _add_memory(self, state, action, reward, new_state, done):
        if len(self.memory) >= self.max_memory_size:
            self.memory.pop(0)
        self.memory.append((state, action, reward, new_state, done))

    def _sample_single_memory(self):
        reward_sum = sum([i[2] for i in self.memory])
        s = random.randint(0, reward_sum)
        probability_sum = 0
        for i in self.memory:
            reward = i[2]
            probability_sum += i[2]/reward_sum
            if probability_sum >= s:
                return i
        return None

    def _sample_memory(self, n):
        if len(self.memory) >= n:
            samples = random.sample(self.memory, n)
            '''
            samples = []
            for i in range(n):
                samples.append(self._sample_single_memory())
            '''
        else:
            samples = []
        return samples

    def _post_run(self):
        W1, W2 = self.sess.run((self.ai.W1, self.ai.W2))
        with open('trained.obj', 'wb') as f:
            pickle.dump((W1, W2), f)
        self.sess.close()

if __name__ == '__main__':
    try:
        n_episodes = int(argv[2])
    except:
        sess = TrainSession()
    else:
        sess = TrainSession(n_episodes=n_episodes)
    finally:
        sess.run()

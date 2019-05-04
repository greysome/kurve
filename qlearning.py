import random
import numpy as np
import tensorflow as tf
from ai import AI

class Memory(object):
    def __init__(self, max_size, alpha):
        self.max_size = max_size
        self.alpha = alpha
        self.buffer = []

    def add(self, state, action, reward, next_state, done, error):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state, done, error))

    def sample(self, n):
        if len(self.buffer) < n:
            return {}

        buffer = sorted(self.buffer, key=lambda x: abs(x[-1]), reverse=True)
        probs = np.array([self.alpha ** i for i in range(len(buffer))])
        probs = probs / sum(probs)
        idxs = np.random.choice(np.arange(len(buffer)),size=n, p=probs)
        samples = {idx: buffer[idx] for idx in idxs}
        return samples

class QLearningSession(object):
    def __init__(self, n_episodes, hidden_neurons,
                 y, K, alpha,
                 learning_rate, sample_n,
                 initial_e, final_e,
                 anneal_frames, observe_frames,
                 update_frames, memory_size):
        self.n_inputs = 30
        self.n_episodes = n_episodes

        self.anneal_frames = anneal_frames
        self.observe_frames = observe_frames
        self.update_frames = update_frames
        self.frames = 0

        self.y = y
        self.e = initial_e
        self.final_e = final_e
        self.K = K
        self.sample_n = sample_n

        self.ai = AI(self.n_inputs,
                     hidden_neurons=hidden_neurons,
                     learning_rate=learning_rate)
        self.memory = Memory(memory_size, alpha)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def run(self):
        for i in range(self.n_episodes):
            state, reward, done = self.env_init()
            actual_actions = pred_actions = ''
            time_alive = 0
            losses = []

            while True:
                time_alive += 1

                action, pred = self._get_action(state)
                pred_actions += str(pred)
                actual_actions += str(action)

                # carry out action `K` times (prevent jerking)
                for j in range(self.K):
                    new_state, reward, done = self.env_step(action)
                    # update memory
                    error = self._get_td_error(state, action, reward, new_state, done)
                    self.memory.add(state, action, reward, new_state, done, error)

                    # update target network
                    if self.frames > self.observe_frames and \
                        self.frames % self.update_frames == 0:
                        print('updating Q- model')
                        W1_assign = self.ai.W1_.assign(self.ai.W1)
                        W2_assign = self.ai.W2_.assign(self.ai.W2)
                        self.sess.run((W1_assign, W2_assign))

                    self.frames += 1
                    self.e -= (self.e-self.final_e) / self.anneal_frames
                    state = new_state

                    if done:
                        break

                # sample memories and train from them
                if self.frames > self.observe_frames:
                    samples = self.memory.sample(self.sample_n)
                    for idx, (*transition, error) in samples.items():
                        self._training_step(*transition)
                        error = self._get_td_error(*transition)
                        # update transition error
                        self.memory.buffer[idx] = (*transition, error)

                loss = self.sess.run(
                    self.ai.loss,
                    feed_dict={self.ai.Q_target:
                               self._get_Q_target(state, action, reward,
                                                  new_state, done),
                               self.ai.inputs: state}
                )
                losses.append(loss)

                if done:
                    print(f'episode {i}')
                    print(f'time alive: {time_alive}')
                    print(f'predicted actions: {pred_actions}')
                    print(f'actual actions: {actual_actions}')
                    print(f'avg loss: {sum(losses)/len(losses)}')
                    print(f'e: {self.e}')
                    print()
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
        Q_target[0, action] = self._get_td_error(state, action, reward, new_state, done)
        return Q_target

    def _get_td_error(self, state, action, reward, new_state, done):
        if done:
            return reward
        else:
            best_action = self.sess.run(self.ai.action,
                                        feed_dict={self.ai.inputs: new_state})
            Q_ = self.sess.run(self.ai.Q_, feed_dict={self.ai.inputs: new_state})
            return reward + self.y*Q_[0, best_action[0]]

    def _training_step(self, state, action, reward, new_state, done):
        Q_target = self._get_Q_target(state, action, reward, new_state, done)
        self.sess.run(self.ai.update_model,
                      feed_dict={self.ai.inputs: state,
                                 self.ai.Q_target: Q_target})
        return Q_target[0, action]

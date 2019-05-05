import random
import numpy as np
import tensorflow as tf
from ai import AI

class Memory(object):
    def __init__(self, max_size, alpha, epsilon):
        self.max_size = max_size
        self.alpha = alpha
        self.epsilon = epsilon
        self.buffer = []

    def add(self, state, action, reward, next_state, done, error):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state, done, error))

    def sample(self, n):
        if len(self.buffer) < n:
            return {}

        # TODO: use sum-tree
        probs = np.array([self._probability(x)**self.alpha for x in self.buffer])
        probs /= sum(probs)
        idxs = np.random.choice(np.arange(len(self.buffer)),size=n, p=probs)
        lol = len([i for i in idxs if self.buffer[i][2] == -1])
        print(lol, n)
        samples = {idx: self.buffer[idx] for idx in idxs}
        return samples

    def _priority(self, transition):
        error = transition[-1]
        return abs(error) + self.epsilon

    def _probability(self, transition):
        error = transition[-1]
        return self._priority(transition)**self.alpha

class QLearningSession(object):
    def __init__(self, n_inputs, n_episodes, hidden_neurons,
                 learning_rate, y, K, sample_n, initial_e, final_e,
                 anneal_frames, observe_frames, update_frames,
                 memory_size, alpha, memory_epsilon):
        self.n_inputs = n_inputs
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
        self.memory = Memory(memory_size, alpha, memory_epsilon)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def run(self):
        for i in range(self.n_episodes):
            state, reward, done = self.env_init()
            actual_actions = pred_actions = ''
            time_alive = 0

            while True:
                time_alive += 1

                action, pred = self._get_action(state)
                pred_actions += str(pred)
                actual_actions += str(action)

                # carry out action `K` times (prevent jerking)
                for j in range(self.K):
                    new_state, reward, done = self.env_step(action)
                    # set maximum error to ensure that
                    # transition is sampled at least once
                    error = 2
                    # update memory
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

                if done:
                    print(f'episode {i}')
                    print(f'time alive: {time_alive}')
                    print(f'predicted actions: {pred_actions}')
                    print(f'actual actions: {actual_actions}')
                    print(f'e: {self.e}')
                    print()
                    break

        self.post_run()

    # To be implemented by subclasses
    def env_init(self):
        pass

    def env_step(self):
        pass

    def post_run(self):
        pass

    def _best_action(self, state):
        return self.sess.run(self.ai.action,
                             feed_dict={self.ai.inputs: state})

    def _Q(self, state):
        return self.sess.run(self.ai.Q,
                             feed_dict={self.ai.inputs: state})

    def _Q_(self, state):
        return self.sess.run(self.ai.Q_,
                             feed_dict={self.ai.inputs: state})

    def _get_action(self, state):
        # return predicted and actual action
        action = self._best_action(state)
        if np.random.rand(1) < self.e:
            return np.random.randint(3), action
        else:
            return action, action

    def _get_Q_target(self, state, action, reward, new_state, done):
        Q_target = self._Q(state)
        if done:
            Q_target[action] = reward
        else:
            new_action = self._best_action(new_state)
            Q_ = self._Q_(new_state)
            Q_target[action] = reward + self.y*Q_[new_action]
        return Q_target

    def _get_td_error(self, state, action, reward, new_state, done):
        Q = self._Q(state)
        if done:
            return reward - Q[action]
        else:
            new_action = self._best_action(new_state)
            Q_ = self._Q_(new_state)
            return reward + self.y*Q_[new_action] - Q[action]

    def _training_step(self, state, action, reward, new_state, done):
        Q_target = self._get_Q_target(state, action, reward, new_state, done)
        self.sess.run(self.ai.update_model,
                      feed_dict={self.ai.inputs: state,
                                 self.ai.Q_target: Q_target})
        return Q_target[action]

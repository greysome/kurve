import pickle
from sys import argv
from engine import Engine
from qlearning import QLearningSession
import matplotlib.pyplot as plt
import config

class TrainSession(QLearningSession):
    def __init__(self, sector_theta, fov, **kwargs):
        n_inputs = fov//sector_theta
        super().__init__(n_inputs=n_inputs, **kwargs)
        self.engine = Engine(1,
                             w=500, h=500,
                             sector_theta=sector_theta, fov=fov)

    def _get_reward_and_done(self):
        if self.engine.players[0].is_alive:
            done = False
            reward = 0
        else:
            done = True
            reward = -1
        return reward, done
        
    def env_init(self):
        self.engine.reset()
        state = self.engine.observe(0)
        reward, done = self._get_reward_and_done()
        return state, reward, done

    def env_step(self, action):
        self.engine.step([(0, action)])
        state = self.engine.observe(0)
        reward, done = self._get_reward_and_done()
        return state, reward, done

    def pre_run(self):
        self.actual_actions = self.predicted_actions = ''
        self.times_alive = []
        self.time_alive = 0
        print(f'running {self.n_episodes} episodes')
        print('-'*30)

    def post_frame(self, actual_action, predicted_action, reward):
        self.time_alive += 1
        self.predicted_actions += str(predicted_action)
        self.actual_actions += str(actual_action)

    def post_episode(self, i):
        print(f'episode {i}')
        print(f'time alive: {self.time_alive}')
        print(f'predicted actions: {self.predicted_actions}')
        print(f'actual actions: {self.actual_actions}')
        print(f'e: {self.e}', end='\n\n')
        self.actual_actions = self.predicted_actions = ''
        self.times_alive.append(self.time_alive)
        self.time_alive = 0

        if i % 5 == 0:
            self.save_model()

    def post_run(self):
        self.save_model()
        self.sess.close()

        # plot time alives
        plt.plot(self.times_alive)
        plt.show()

    def save_model(self):
        W1, W2 = self.sess.run((self.ai.W1, self.ai.W2))
        with open('trained.model', 'wb') as f:
            pickle.dump((W1, W2), f)
        print('-'*10, 'saving model in trained.model', '-'*10)

if __name__ == '__main__':
    # retrieve only user-defined variables
    kwargs = {k: v for k, v in config.__dict__.items() \
              if not k.startswith('__')}
    sess = TrainSession(**kwargs)
    sess.run()

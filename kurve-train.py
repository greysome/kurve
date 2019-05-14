import pickle
from sys import argv
from engine import Engine
from qlearning import QLearningSession
import matplotlib.pyplot as plt
import config

class TrainSession(QLearningSession):
    def __init__(self, model_file, sector_theta, fov, **kwargs):
        n_inputs = fov//sector_theta
        super().__init__(n_inputs=n_inputs, **kwargs)
        self.model_file = model_file
        self.load_model()
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
        self.print_header(f'running {self.n_episodes} episodes')
        self.actual_actions = self.predicted_actions = ''
        self.times_alive = []
        self.time_alive = 0

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

        if i % 50 == 0:
            self.save_model()

    def post_run(self):
        self.save_model()
        self.sess.close()

        # plot time alives
        plt.plot(self.times_alive)
        plt.show()

    def load_model(self):
        with open(self.model_file, 'rb') as f:
            W1, W2 = pickle.load(f)
            W1_assign = self.ai.W1.assign(W1)
            W2_assign = self.ai.W2.assign(W2)
            W1__assign = self.ai.W1_.assign(W1)
            W2__assign = self.ai.W2_.assign(W2)
            self.sess.run((W1_assign, W2_assign, W1__assign, W2__assign))
        self.print_header(f'loaded {self.model_file}')

    def save_model(self):
        W1, W2 = self.sess.run((self.ai.W1, self.ai.W2))
        with open('trained.model', 'wb') as f:
            pickle.dump((W1, W2), f)
        self.print_header('saving model in trained.model')

if __name__ == '__main__':
    model_name = argv[1]
    # retrieve only user-defined variables
    kwargs = {k: v for k, v in config.__dict__.items() \
              if not k.startswith('__')}
    sess = TrainSession(**kwargs,
                        model_file=model_name+'.model',
                        n_actions=3)
    sess.run()

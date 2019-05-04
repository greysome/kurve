import pickle
from sys import argv
from engine import Engine
from qlearning import QLearningSession
import config

class TrainSession(QLearningSession):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.engine = Engine(1)

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

    def post_run(self):
        W1, W2 = self.sess.run((self.ai.W1, self.ai.W2))
        with open('trained.model', 'wb') as f:
            pickle.dump((W1, W2), f)
        print('trained model saved in trained.model')
        self.sess.close()

if __name__ == '__main__':
    # retrieve only user-defined variables
    kwargs = {k: v for k, v in config.__dict__.items() \
              if not k.startswith('__')}
    sess = TrainSession(**kwargs)
    sess.run()

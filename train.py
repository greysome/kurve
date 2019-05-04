import pickle
from sys import argv
from engine import Engine
from qlearning import QLearningSession
import config

class TrainSession(QLearningSession):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.engine = Engine(1)
        
    def env_init(self):
        self.engine.reset()
        return self.engine.observe(0)

    def env_step(self, action):
        self.engine.step([(0, action)])
        return self.engine.observe(0)

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

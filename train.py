import pickle
from sys import argv
from engine import Engine
from qlearning import QLearningSession
from config import *

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
    sess = TrainSession(n_episodes=n_episodes,
                        hidden_neurons=hidden_neurons,
                        y=y, K=K, learning_rate=learning_rate,
                        initial_e=initial_e, final_e=final_e,
                        anneal_frames=anneal_frames, observe_frames=observe_frames,
                        update_frames=update_frames, memory_size=memory_size):
    sess.run()

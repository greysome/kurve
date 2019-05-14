from sys import argv
import gym
import matplotlib.pyplot as plt
from qlearning import QLearningSession

class EnvTrainSession(QLearningSession):
    def __init__(self, env_name):
        self.rewards = []
        self.reward = 0

        self.env = gym.make(env_name)
        super().__init__(n_inputs=self.env.observation_space.shape[0],
                         n_actions=self.env.action_space.n,
                         n_episodes=200,
                         hidden_neurons=4,
                         observe_frames=0,
                         anneal_frames=5000)

        '''
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        plt.ion()
        self.fig.show()
        '''

    def env_init(self):
        state = self.env.reset()
        return state, 0, False

    def env_step(self, action):
        self.env.render()
        state, reward, done, _ = self.env.step(action)
        return state, reward, done

    def post_frame(self, actual_action, predicted_action, reward):
        self.reward += reward

    def post_episode(self, i):
        print(self.sess.run(self.ai.W1)[0,0])
        print(f'episode {i}')
        print(f'reward: {self.reward}')
        print(f'e: {self.e}', end='\n\n')
        self.rewards.append(self.reward)
        self.reward = 0

        '''
        self.ax.clear()
        self.ax.plot(self.rewards)
        self.fig.canvas.draw()
        '''

    def post_run(self):
        plt.plot(self.rewards)
        plt.show()

if __name__ == '__main__':
    env_name = argv[1]
    if env_name == 'l':
        for i in gym.envs.registry.all():
            print(i)
    else:
        session = EnvTrainSession(env_name)
        session.run()

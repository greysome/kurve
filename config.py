# how many games to run total?
n_episodes = 500

# no. of neurons in hidden layer
hidden_neurons = 20

# future reward discount
y = 0.9

# sample memories every `K` frames
K = 10

# how much extra weight to give to unusual experiences
# (i.e. experiences with the greaest absolute reward)?
# 1 -> none, 0 -> only the most unusual experiences are
# sampled
alpha = 0.99

# how many memories to sample each training step?
sample_n = 10

# learning rate of gradient descent
learning_rate = 0.01

# exploration-exploitation factor
# every frame, a random action will be sampled with probability e
initial_e = 1
# e value to arrive at after `anneal_frames` frames
final_e = 0.01
anneal_frames = 10000

# number of frames to collect memories before training on them
observe_frames = 1000

# update target network every `update_frames` frames
update_frames = 300

# maximum memory size
memory_size = 5000

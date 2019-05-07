#---------- ENGINE ----------#
# field of vision, in degrees
fov = 150

# how big each sector is, in degrees
sector_theta = 5

#---------- AI ----------#
# no. of neurons in hidden layer
hidden_neurons = 20

#---------- Q-LEARNING ----------#
# how many games to run total?
n_episodes = 60

# future reward discount
y = 0.9

# carry out the same action for `K` frames
K = 10

# how many memories to sample each training step?
sample_n = 30

# learning rate of gradient descent
learning_rate = 0.01

# exploration-exploitation factor
# every frame, a random action will be sampled with probability e
initial_e = 1

# e value to arrive at after `anneal_frames` frames
final_e = 0.01

# number of frames over which to anneal the value of
# exploration-explotation factor to `final_e`
anneal_frames = 50000

# number of frames to collect memories before training on them
observe_frames = 1000

# update target network every `update_frames` frames
update_frames = 1000

#---------- MEMORY ----------#
# maximum memory size
memory_size = 5000

# how much extra weight to give to unusual experiences
# (i.e. experiences with the greaest absolute reward)?
# 1 -> all experiences given equal weight
# 0 -> only the most unusual experiences are sampled
alpha = 0.99

# added to TD-error to ensure non-zero priority (and probability)
memory_epsilon = 0.01

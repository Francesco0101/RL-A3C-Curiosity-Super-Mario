WORLD = 1
STAGE = 1
LR = 1e-4
GAMMA = 0.9
TAU = 1.0
ENTROPY_COEFF = 0.01
VALUE_LOSS_COEF = 0.5
MAX_GRAD_NORM = 250
NUM_LOCAL_STEPS = 50
NUM_GLOBAL_STEPS = 5e6
MAX_EPISODES = 100000
NUM_WORKERS = 12
ARGMAX_WORKERS = 2
SAVE_EPISODE_INTERVAL = 1000
LOG_EPISODE_INTERVAL = 100
LOG_PATH = "log/"
SAVE_PATH = "checkpoints/"
LAMBDA = 1.0
ETA = 0.2
BETA = 0.2
REWARD_TYPE = "sparse"
WEIGHT_CURIOSITY = 10.0
SHARED_OPTIMIZER = False

import numpy as np
import gym
from gym.utils import seeding
from gym.spaces import Space, Discrete, Box

NB_STEPS = 10000
INPUT_SIZE = 3
VERBOSE = True

def log(*args):
    if VERBOSE:
        print(*args)

def prepro_state(raw_state):
    # maximum=10.
    # minimum=-10
    # return (raw_state-minimum)/(maximum-minimum)
    return raw_state

def prepro_reward(raw_reward):
    prep_reward = raw_reward
    return prep_reward

def inv_prepro_reward(prep_action):
    raw_action = prep_action
    return raw_action

class EnvDumb(gym.Env):
    def __init__(self, env_config, seed=42):
        np.random.seed(seed)

        # Domain specific variables
        self.round = 0
        self.TRAIN = True
        self.action_type = env_config["action_type"]  # "continuous" or "discrete"

        self.init_state = np.array([5.0 for i in range(INPUT_SIZE)], dtype=np.float32)
        self.init_reward = 0
        self.init_done = False
        self.init_info = {}
        self.init_round = 0

        # observation space
        self.Ns = INPUT_SIZE
        self.observation_space = Box(low=-11., high=11., shape=(self.Ns,), dtype=np.float32)

        # Action space
        self.Na = INPUT_SIZE
        if self.action_type == "discrete":
            self.action_space = Discrete(self.Na * 2)
        elif self.action_type == "continuous":
            self.action_space = Box(-1., 1., (self.Na,))
        else:
            raise ValueError("action type not understood. Should be in {discrete,continous}")
        self.metadata = {"render.modes": ["human"]}

        # Set with initial state
        # "state" mentions the raw state
        self.state, self.reward, self.done, self.info, self.round = None, None, None, None, None

        self.reset(testing=False)

        try:
            assert (self.observation_space.contains(self.state))
        except AssertionError:
            log("ERROR : INVALID INIT STATE (bounds or type)", self.state)

    def get_reward(self):
        maxi = np.sum(np.array([10 for i in range(INPUT_SIZE)]) ** 2)
        mini = 0
        return 1 - (np.sum(self.state ** 2) - mini) / (maxi - mini)

    def step(self, action):
        try:
            assert (self.action_space.contains(action))
        except AssertionError:
            log("WARNING : INVALID ACTION", action)

        # Update state
        self.round += 1
        if self.action_type == "discrete":
            mapping = {0: [-1, 0, 0], 1: [1, 0, 0], 2: [0, -1, 0], 3: [0, 1, 0], 4: [0, 0, -1], 5: [0, 0, 1]}
            self.state = self.state + np.array(mapping[ action ], dtype=np.float32)
        elif self.action_type == "continuous":
            self.state = self.state + action
        self.reward = self.get_reward()  # reward before clipping before improve penalty to the edges
        self.state = np.clip(self.state, -10., 10.)  # colision
        self.done = self.round >= NB_STEPS

        # Check state
        try:
            assert (self.observation_space.contains(self.state))
        except AssertionError:
            log("WARNING : INVALID STATE", action)

        # Prepo for ML algo.
        prep_state = prepro_state(self.state)
        prep_reward = prepro_reward(self.reward)
        return prep_state, prep_reward, self.done, self.info

    def reset(self, testing=False):
        self.round = self.init_round
        self.reward = self.init_reward
        self.done = self.init_done
        self.info = self.init_info
        self.state = self.init_state
        return self.state

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

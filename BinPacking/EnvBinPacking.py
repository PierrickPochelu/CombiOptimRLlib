import numpy as np
import gym
from gym.utils import seeding
from gym.spaces import Space, Discrete, Box

# For PoC
# ITEMS=[0.1, 0.2, 0.5, 1., 0.1, 0.2, 0.5, 0.1, 0.2, 0.1]
# BINS = [1, 1, 1]

VERBOSE = False

# https://epub.jku.at/obvulihs/download/pdf/6996324?originalFilename=true

NB_ITEMS = 100
ITEMS = []
for i in range(NB_ITEMS // 2):
    ITEMS.append(2.)
for i in range(NB_ITEMS // 2):
    ITEMS.append(3.)
NB_MAXIMUM_BIN_LEVEL = 9

"""
NB_ITEMS = 100
ITEMS = [1, 2, 3, 4]
for j in range(1, 9):
    for i in range(12): # 8*12=96
        ITEMS.append(j)

NB_ITEMS = 100
ITEMS = []
for j in range(1, 5):
    for i in range(25):
        ITEMS.append(j)
assert(NB_ITEMS==len(ITEMS))
NB_MAXIMUM_BIN_LEVEL = 9

"""

FREQ_BINS = [0 for i in range(NB_MAXIMUM_BIN_LEVEL)]

def log(*args):
    if VERBOSE:
        print(*args)

def inv_prepro_state(prep_state):
    raw_state = prep_state.copy()
    raw_state[:-1] *= float(NB_ITEMS)
    raw_state[-1] *= 3.
    return raw_state
def prepro_state(raw_state):
    prep_state = raw_state.copy()
    prep_state[:-1] /= float(NB_ITEMS)
    prep_state[-1] /= 3.
    return prep_state


def prepro_reward(raw_reward):
    prep_reward = raw_reward
    return prep_reward


def inv_prepro_reward(prep_action):
    raw_action = prep_action
    return raw_action


class EnvBinPacking(gym.Env):
    def __init__(self, env_config, seed=42):
        # Domain specific variables
        self.action_type = env_config["action_type"]  # "continuous" or "discrete"

        # Init values
        self.init_items = ITEMS.copy()
        self.init_state = np.zeros(len(FREQ_BINS)+1, dtype=np.float32)  # Frequence-1 + next element
        self.init_state[:len(FREQ_BINS)] = FREQ_BINS
        self.init_state[-1] = self.init_items[-1]

        self.init_reward = 0
        self.init_done = False
        self.init_info = {}
        self.init_round = 0

        # Variable values (in addition to
        self.cur_items = None
        # "state" mentions the raw state
        self.state, self.reward, self.done, self.info = None, None, None, None

        # observation space
        self.Ns = len(FREQ_BINS)+1  # Bins level from 0 to `NB_MAXIMUM_BIN_LEVEL` + the current item to place
        # nota bene: the fully filled bin is not modelized
        self.observation_space = Box(low=0, high=len(ITEMS), shape=(self.Ns,), dtype=np.float32)

        # Action space
        self.Na = len(FREQ_BINS)  # Bins level from 0 (opening a new one) to `NB_MAXIMUM_BIN_LEVEL`
        if self.action_type == "discrete":
            self.action_space = Discrete(self.Na)
        elif self.action_type == "continuous":
            self.action_space = Box(-1., 1., (self.Na,))
        else:
            raise ValueError("action type not understood. Should be in {discrete,continous}")
        self.metadata = {"render.modes": ["human"]}

        self.reset(testing=False)

        try:
            assert (self.observation_space.contains(self.state))
        except AssertionError:
            log("ERROR : INVALID INIT STATE (bounds or type)", self.state)

    def step(self, action):
        try:
            assert (self.action_space.contains(action))
        except AssertionError:
            log("WARNING : INVALID ACTION", action)

        prev_state = self.state.copy()  # for debugging

        self.state, self.reward, self.done, self.info = self._transition(action)

        # Check state
        try:
            assert (self.observation_space.contains(self.state))
        except AssertionError:
            log("WARNING : INVALID STATE", self.state)

        # Returns info for ML algo.
        out = (prepro_state(self.state), prepro_reward(self.reward), self.done, self.info)

        # Reset for next training iterations
        log(f"action:{action} "
            f"reward:{self.reward} "
            f"prev_state:{prev_state} -> new_state:{self.state} "
            f"{self.done}")

        return out

    def _transition(self, action):  # action -> raw_state, raw_reward, done, info
        assert (self.action_type == "discrete")

        incoming_item_size = self.cur_items.pop(-1)  # 2 or 3
        old_bin_size = action
        bin_space_before = NB_MAXIMUM_BIN_LEVEL - old_bin_size
        bin_space_after = bin_space_before - incoming_item_size
        new_bin_size = incoming_item_size + old_bin_size

        # Constraint: overflow!
        if bin_space_after < 0:
            self.reward = -1000 + bin_space_after
            self.done = True
            return self.state, self.reward, self.done, self.info

        # Constraint: logical constraint: we cannot put it into a bin at `action` because there is no bin like this
        if old_bin_size != 0:
            if self.state[old_bin_size] == 0:
                self.reward = -1000
                self.done = True
                return self.state, self.reward, self.done, self.info

        # Update state
        if new_bin_size == NB_MAXIMUM_BIN_LEVEL:
            self.state[int(old_bin_size)] -= 1. # This bin is removed from the observable state
        elif new_bin_size < NB_MAXIMUM_BIN_LEVEL:
            self.state[int(new_bin_size)] += 1.
            if int(old_bin_size)>0: # bin "0" is a generator
                self.state[int(old_bin_size)] -= 1. # update bin value
            else:
                pass # a new bin is open
        else:
            raise ValueError("Not expected overflow")

        # reward
        #self.reward = -bin_space_after # incremental waste
        self.reward = 0 if int(old_bin_size)>0 else -bin_space_after #-1 if bin is open, 0 otherwise

        # next item if any, other
        if self.cur_items:
            self.state[-1] = self.cur_items[-1]
        else:
            self.state[-1] = 0  # default value
            self.done = True  # last value

        return self.state, self.reward, self.done, self.info

    def reset(self, testing=False):


        self.reward = self.init_reward
        self.done = self.init_done
        self.info = self.init_info.copy()
        self.state = self.init_state.copy()

        self.cur_items = self.init_items.copy()

        return self.state

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

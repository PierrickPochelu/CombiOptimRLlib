import numpy as np
from ray.tune import Trainable
from EnvBinPacking import NB_MAXIMUM_BIN_LEVEL, inv_prepro_reward, inv_prepro_state

VERBOSE=False
def log(*args):
    if VERBOSE:
        print(*args)

class AgentBestFit(Trainable):
    def __init__(self, trainer, env_class, env_config):
        self.env_class = env_class
        self.env_config = env_config
        self.env = self.env_class(self.env_config)
        self.trainer = None  # No smart algorithm

    def train(self):
        self.reset(False)
        pass  # no need to be trained

    def reset(self, testing=False):
        # self.env = self.env_class(self.env_config)
        obs = self.env.reset(testing=testing)
        return obs, self.env

    def evaluate(self):
        prep_obs, env = self.reset(testing=True)
        cumulated_rewards = 0
        done = False
        while not done:
            action = self.trainer_prediction(prep_obs)
            prep_obs, prep_reward, done, info = env.step(action)
            raw_reward = inv_prepro_reward(prep_reward)
            raw_obs = inv_prepro_state(prep_obs)
            cumulated_rewards += raw_reward
        return {"cumulated_rewards": cumulated_rewards, "last_reward": raw_reward, "last_state": raw_obs,
                "last_action": action}

    def trainer_prediction(self, prepro_obs):
        raw_obs=inv_prepro_state(prepro_obs)
        bin_freqs=raw_obs[:-1]
        next_item=raw_obs[-1]

        # Parse already opened bin
        for i in range(len(bin_freqs)-1,-1,-1): # from bigger to smaller  open bins
            if bin_freqs[i]>0:
                prev_bin_level=i
                new_bin_level=prev_bin_level+next_item

                if new_bin_level <= NB_MAXIMUM_BIN_LEVEL:
                    return i

        return 0 # A new bin is open
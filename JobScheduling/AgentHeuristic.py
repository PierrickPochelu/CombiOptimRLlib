import numpy as np
from ray.tune import Trainable


VERBOSE=False
def log(*args):
    if VERBOSE:
        print(*args)

class AgentFCFS_FCFS(Trainable):
    def __init__(self, trainer, env_class, env_config, inv_prepro_state, inv_prepro_reward):
        self.env_class = env_class
        self.env_config = env_config
        self.env = self.env_class(self.env_config)
        self.trainer = None  # No smart algorithm
        self.inv_prepro_state=inv_prepro_state
        self.inv_prepro_reward=inv_prepro_reward

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
        info=env.info

        while not done:
            action = self.trainer_prediction(env, info)
            raw_obs, prep_reward, done, info = env.step(action)

            raw_reward = self.inv_prepro_reward(prep_reward)
            prep_obs = self.inv_prepro_state(raw_obs)
            cumulated_rewards += raw_reward
        return {"cumulated_rewards": cumulated_rewards, "last_reward": raw_reward, "last_state": prep_obs,
                "last_action": action}

    def trainer_prediction(self, env, info):
        I=info["incoming_item"]
        r=env.round

        if I is None:
            return 6 # older of the list
        elif self.env.ss.can_fit( r , I.weight ):
            return 1 # current item
        else:
            return 6 # older of the list


class AgentFCFS_EASY_BACKFILLING(Trainable):
    def __init__(self, trainer, env_class, env_config, inv_prepro_state, inv_prepro_reward):
        self.env_class = env_class
        self.env_config = env_config
        self.env = self.env_class(self.env_config)
        self.trainer = None  # No smart algorithm
        self.inv_prepro_state = inv_prepro_state
        self.inv_prepro_reward = inv_prepro_reward

    def train(self):
        self.reset(False)
        pass  # no need to be trained

    def reset(self, testing=False):
        # self.env = self.env_class(self.env_config)
        obs = self.env.reset(testing=testing)
        return obs, self.env

    def evaluate(self):
        raw_obs, env = self.reset(testing=True)
        prep_obs = self.inv_prepro_state(raw_obs)
        cumulated_rewards = 0
        done = False
        info = env.info

        while not done:
            action = self.trainer_prediction(env, info)
            raw_obs, prep_reward, done, info = env.step(action)

            raw_reward = self.inv_prepro_reward(prep_reward)
            raw_obs = self.inv_prepro_state(raw_obs)
            cumulated_rewards += raw_reward
        return {"cumulated_rewards": cumulated_rewards, "last_reward": raw_reward, "last_state": raw_obs,
                "last_action": action}

    def trainer_prediction(self, env, info):

        I = info["incoming_item"]
        r = env.round

        if I is None:
            return 2
        elif self.env.ss.can_fit(r, I.weight):
            return 1  # current item
        else:
            return 2


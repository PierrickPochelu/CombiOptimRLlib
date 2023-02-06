import numpy as np
from ray.tune import Trainable
from ray.rllib.algorithms import dqn
from typing import *

VERBOSE=False
def log(*args):
    if VERBOSE:
        print(*args)

def identity(x):
    return x

class AgentRLLIB():
    def __init__(self, trainer,
                 env_class,
                 env_config,
                 inv_prepro_state:Callable=identity,
                 inv_prepro_reward:Callable=identity):
        self.env_class=env_class
        self.env_config=env_config

        self.inv_prepro_state=inv_prepro_state
        self.inv_prepro_reward=inv_prepro_reward
        self.env_config["horizon"]=None
        self.env_config["soft_horizon"]=True
        self.env=self.env_class(self.env_config)
        self.trainer=trainer

    def train(self):
        self.reset(False)
        self.trainer.train()

    def reset(self, testing=False):
        raw_obs = self.env.reset(testing=testing)
        return raw_obs, self.env

    def evaluate(self):
        raw_obs, env = self.reset(testing=True)
        prep_obs=self.inv_prepro_state(raw_obs)
        cumulated_rewards = 0
        done = False

        while not done:
            action, prob = self.trainer_prediction(self.trainer, prep_obs)
            prep_obs, prep_reward, done, info = env.step(action)
            raw_reward = self.inv_prepro_reward(prep_reward)
            raw_obs = self.inv_prepro_state(prep_obs)

            cumulated_rewards += raw_reward

        return {"cumulated_rewards":cumulated_rewards,
                "last_reward":raw_reward,
                "last_state":raw_obs,
                "last_action":action,
                "done":done,
                "info":str(info)}

    def save(self, checkpoint_dir):
        file_path=self.trainer.save(checkpoint_dir)
        return file_path

    def restore(self, path):
        self.trainer.restore(path)

    def trainer_prediction(self, trainer, observation, stochasticity=False):
        """
        :param trainer: RLlib algorithm
        :param observation: processed state for the RL algorithm
        :param stochasticity: keeps it false
        :return: If action space is discrete it returns action as int and probability vector.
        If action space is continuous it returns two time the prediction
        """
        agent = trainer.workers._local_worker

        policy_name = "default_policy"
        clip_action = agent.policy_config["clip_actions"]
        preprocessed = agent.preprocessors[policy_name].transform(observation)
        filtered_obs = agent.filters[policy_name](preprocessed, update=False)
        policy = agent.get_policy(policy_name)
        result = policy.compute_single_action(filtered_obs)
        # older RLlib version
        #result=policy.compute_single_action(filtered_obs, state, None, None, None, clip_actions=clip_action)

        if isinstance(trainer, dqn.DQN):
            proba_name = "q_values"
            logit = result[2][proba_name]
            policy_pred = self.__softmax(logit)
            if stochasticity:
                action = result[0]
            else:
                action = np.argmax(policy_pred)
        else:
            if self.env_config["action_type"]=="discrete":
                action=result[0]
                n=policy.action_space.n
                # probs is one-hot encoded
                policy_pred=np.zeros((n,),dtype=np.float32)
                policy_pred[action]=1.
            else:
                action=result[0]
                policy_pred=action
        return action, policy_pred

    def __filter_nan(self,x):
        if np.sum(np.isnan(x))>0:
            return np.random.uniform(0,1,(x.shape))
        else:
            return x

    def __softmax(self,x):
        max_x=np.max(x)
        min_x=np.min(x)
        x01=(x-min_x)/(max_x-min_x+1e-7)
        exp_x=np.exp(x01)
        y=exp_x / (np.sum(exp_x)+1e-7)
        return y

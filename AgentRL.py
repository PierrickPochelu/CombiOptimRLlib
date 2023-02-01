import numpy as np
from ray.tune import Trainable
from ray.rllib.algorithms import dqn
from EnvBinPacking import inv_prepro_state, inv_prepro_reward

VERBOSE=False
def log(*args):
    if VERBOSE:
        print(*args)

class AgentRLLIB(Trainable):
    def __init__(self, trainer, env_class, env_config):
        self.env_class=env_class
        self.env_config=env_config

        self.env_config["horizon"]=None
        self.env_config["soft_horizon"]=True
        self.env=self.env_class(self.env_config)
        self.trainer=trainer

    def train(self):
        self.reset(False)
        self.trainer.train()

    def reset(self, testing=False):
        # self.env = self.env_class(self.env_config)
        obs = self.env.reset(testing=testing)
        return obs, self.env

    def evaluate(self):
        prep_obs, env = self.reset(testing=True)
        prev_raw_obs = inv_prepro_state(prep_obs)
        cumulated_rewards = 0
        done = False
        while not done:
            action, prob = self.trainer_prediction(self.trainer, prep_obs)
            prep_obs, prep_reward, done, info = env.step(action)
            raw_reward = inv_prepro_reward(prep_reward)
            raw_obs = inv_prepro_state(prep_obs)

            cumulated_rewards += raw_reward

            log(f"action:{action} "
                  f"reward:{raw_reward} "
                  f"prev_state:{prev_raw_obs} -> new_state:{raw_obs} "
                  f"{done}")

            prev_raw_obs=raw_obs
        return {"cumulated_rewards":cumulated_rewards,
                "last_reward":raw_reward,
                "last_state":prep_obs,
                "last_action":action,
                "done":done}

    def _save(self, checkpoint_dir):
        print("SAVE")
        file_path = checkpoint_dir
        self.trainer.save(file_path)
        print("SAVE END")
        return file_path

    def _restore(self, path):
        print("RESTORE")
        checkpoint_leaf_path=self.get_restore_path(path)
        self.trainer.restore(checkpoint_leaf_path)
        print("RESTORE END")

    def trainer_prediction(self, trainer, observation, stochasticity=False):
        agent = trainer.workers._local_worker
        proba_name = "q_values"  # "logits"

        policy_name = "default_policy"
        clip_action = agent.policy_config["clip_actions"]
        preprocessed = agent.preprocessors[policy_name].transform(observation)
        filtered_obs = agent.filters[policy_name](preprocessed, update=False)
        policy = agent.get_policy(policy_name)
        result = policy.compute_single_action(filtered_obs)
        #result = policy.compute_single_action(filtered_obs, state)
        #result=policy.compute_single_action(filtered_obs, state, None, None, None, clip_actions=clip_action)
        #result = trainer._policy.compute_single_action(filtered_obs, state, None, None, None, clip_actions=clip_action)
        #result = trainer.compute_action(observation)

        if isinstance(trainer, dqn.DQN):
            raw_probs = result[2][proba_name]
            probs = self.__softmax(raw_probs)
            if stochasticity:
                action = result[0]
            else:
                action = np.argmax(probs)
        else:
            if self.env_config["action_type"]=="discrete":
                action=result[0]
                n=policy.action_space.n
                # probs is one-hot encoded
                probs=np.zeros((n,),dtype=np.float32)
                probs[action]=1.
            else:
                action=result[0]
                action=self.__filter_nan(action)
                probs=action
        return action, probs

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

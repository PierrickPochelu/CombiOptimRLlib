import numpy as np
from ray.tune import Trainable
import os

def apply_continuous_combiner(actions):
    """
    Compute the closest action of the centroid. Pierrick Pochelu's thesis.
    :param actions:
    :return:
    """
    # centroid
    centroid = np.average(actions, axis=0)

    # compute distance
    distances = []
    for a in actions:
        dist = np.sum(np.abs(a - centroid))  # absolute distance
        distances.append(dist)

    # action selection
    selected_action_id = np.argmin(distances)

    return centroid #actions[selected_action_id]

def apply_discrete_combiner(actions):
    """
    Averaging of probs
    :param actions:
    :return:
    """
    return np.average(actions, axis=0)


class EnsembleAgentRL(Trainable):
    def __init__(self, trainable_applications):
        self.NB_MODELS=len(trainable_applications)
        self.trainable_applications = trainable_applications

    def trainer_prediction(self, ob):
        # All base models predict
        probs=[]
        for app in self.trainable_applications:
            _, prob = app.trainer_prediction(app.trainer, ob)
            probs.append(prob)

         # Combine prediction
        if self.trainable_applications.env_config["action_type"]=="discrete":
            action=apply_discrete_combiner(probs)
        elif self.trainable_applications.env_config["action_type"]=="continuous":
            action=apply_continuous_combiner(probs)
        else:
            raise ValueError("unexpected action type")
        return action

    def train(self):
        # Train the population
        infos = []
        for app in self.trainable_applications:
            info = app.train()
            infos.append(info)
        return infos

    def evaluate(self):
        envs = [app.env_class for app in self.trainable_applications]
        env_configs = [app.env_config for app in self.trainable_applications]
        app0=self.trainable_applications[0]
        is_discrete = env_configs[0]["action_type"] == "discrete"
        env_callable = envs[0]
        env = env_callable(env_configs[0])

        raw_obs = env.reset(testing=True)
        prep_obs=app0.inv_prepro_state(raw_obs)

        cumulated_reward = 0
        done = False

        while not done:
            if is_discrete:
                action_probs = self.trainer_prediction(prep_obs)
                action = np.argmax(action_probs)
            else:
                action = self.trainer_prediction(prep_obs)
                action = np.minimum(action, np.ones(action.shape)) # clipping

            raw_obs, raw_reward, done, info = env.step(action)
            prep_obs=app0.inv_prepro_state(raw_obs)
            prep_reward=app0.inv_prepro_reward(raw_reward)

            cumulated_reward += raw_reward

        env.reset(testing=True)
        return {"cumulated_rewards":cumulated_reward,
                "last_reward":raw_reward,
                "last_state":raw_obs,
                "last_action":action,
                "done":done}


import os
import sys
import numpy as np

from rl_factory import rl_agent_factory  # Factory to build RL agents name:stirng->model:rllib
from rl_factory import rl_hyperparameter_space_dict_factory
from BinPacking.EnvBinPacking import EnvBinPacking, inv_prepro_state, inv_prepro_reward
from AgentRL import AgentRLLIB
from EnsembleAgentRL import EnsembleAgentRL

if __name__ == "__main__":
    # build the environment
    env_config = {"action_type": "discrete"}
    env_class = EnvBinPacking

    # build the RL agent
    rl_name = "PPO"  # PPO implementation fits {"continuous" or "discrete"}. DQN "discrete". DDPG "continuous".

    # e.g. name:string -> {"lr":[0.1,0.01,...], "nb_layers":[1, 2, ...], ...}
    hyperparam_space_default_value, _ = rl_hyperparameter_space_dict_factory(rl_name)

    hyperparameters = hyperparam_space_default_value.copy()
    for k, v in hyperparam_space_default_value.items():
        hyperparameters[k] = v[len(v) // 2]  # common values are at the center

    # update value similar to the publication
    hyperparameters["lr"] = 1e-4
    hyperparameters["deep"] = 2
    hyperparameters["wide"] = 16
    hyperparameters["train_batch_size"] = 64
    hyperparameters["sgd_minibatch_size"] = 64
    hyperparameters["lambda"] = 0.99
    rllib_trainer = rl_agent_factory(rl_name, hyperparameters, env_class, env_config=env_config)

    nb_agents = 4
    episodes = 200
    eval_every = 100
    eval_times = 30

    AGENTS_INFO = dict()  # {id -> {"path"->path:str, "score"->score:int, "history"->hist:List[int]}  }

    for a in range(nb_agents):
        agent = AgentRLLIB(rllib_trainer, env_class, env_config,
                           inv_prepro_state=inv_prepro_state, inv_prepro_reward=inv_prepro_reward)

        agent_info = {"folder": "/tmp/ensemble/agent" + str(a) + "/",
                      "checkpoint_path": "undefined",
                      "score": -np.inf,
                      "history": []}

        for i in range(episodes):
            agent.train()

            if i % eval_every == 0:
                score = np.round(np.mean([agent.evaluate()["cumulated_rewards"] for i in range(eval_times)]), 2)
                agent_info["history"].append(score)

                # better score found
                if score >= agent_info["score"]:
                    agent_info["score"] = score
                    agent_info["checkpoint_path"] = agent.save(agent_info["folder"])
                print("Score:", score)

        AGENTS_INFO[a] = agent_info

    print("BEST AGENT SUMMARY:")
    print(AGENTS_INFO)


    ensemble = []
    for agent_info in AGENTS_INFO.values():
        # build an empty RL agent
        agent = AgentRLLIB(rllib_trainer, env_class, env_config,
                           inv_prepro_state=inv_prepro_state, inv_prepro_reward=inv_prepro_reward)

        # restore weights
        agent.restore(agent_info["checkpoint_path"])

        # save it
        ensemble.append(agent)

    ensemble = EnsembleAgentRL(ensemble)
    score = np.round(np.mean([ensemble.evaluate()["cumulated_rewards"] for i in range(eval_times)]), 2)
    print("Ensemble score:", score)

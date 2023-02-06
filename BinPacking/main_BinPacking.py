from rl_factory import rl_agent_factory # Factory to build RL agents name:stirng->model:rllib
from rl_factory import rl_hyperparameter_space_dict_factory
from BinPacking.EnvBinPacking import EnvBinPacking, inv_prepro_state, inv_prepro_reward
from AgentRL import AgentRLLIB
from BinPacking.AgentHeuristic import AgentBestFit

# Online Bin Packing formulated as: https://epub.jku.at/obvulihs/download/pdf/6996324?originalFilename=true
# In short:
# Input space: [B_0, B_1, B_i, .. B_n-1, I] # B_i contains number of bins. The last value `I` is the incoming item size.
# Reward: espace left after puting `I` item (negative value to maximize). Masking due to logical/constraints (e.g. bin overflow)
# Action space: Bin-level from 0 to n-1.

# By changing EnvBinPacking, this project allows faster adaptation to real wold cases (chipset layout design, HPC job scheduling, DAG partioning...)
# Changing `rl_name` allows fast changing of RL algorithm

if __name__=="__main__":
    # build the environment
    env_config = {"action_type": "discrete"}
    env_class=EnvBinPacking

    # build the RL agent
    rl_name = "PPO" # PPO implementation fits {"continuous" or "discrete"}. DQN "discrete". DDPG "continuous".

    # e.g. name:string -> {"lr":[0.1,0.01,...], "nb_layers":[1, 2, ...], ...}
    hyperparam_space_default_value, _ = rl_hyperparameter_space_dict_factory(rl_name)

    hyperparameters = hyperparam_space_default_value.copy()
    for k,v in hyperparam_space_default_value.items():
        hyperparameters[k]=v[len(v)//2] # common values are at the center

    # update value similar to the publication
    hyperparameters["lr"]=3e-4
    hyperparameters["deep"]=2
    hyperparameters["wide"]=16
    hyperparameters["train_batch_size"]=64
    hyperparameters["sgd_minibatch_size"]=64
    hyperparameters["lambda"]=0.99
    rllib_trainer = rl_agent_factory(rl_name, hyperparameters, env_class, env_config=env_config)


    agent=AgentRLLIB(rllib_trainer, env_class, env_config, inv_prepro_state, inv_prepro_reward)
    episodes=5
    for i in range(episodes):
        for i in range(5):
            agent.train()

        score=agent.evaluate()
        print("Score:", score)



    # Basic
    agent = AgentBestFit(None, env_class, env_config, inv_prepro_state, inv_prepro_reward)
    score = agent.evaluate()
    print("Score:", score)

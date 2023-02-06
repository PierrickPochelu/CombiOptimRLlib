from rl_factory import rl_agent_factory # Factory to build RL agents name:stirng->model:rllib
from rl_factory import rl_hyperparameter_space_dict_factory
from JobScheduling.EnvScheduling import EnvScheduling, inv_prepro_state, inv_prepro_reward
from AgentRL import AgentRLLIB

if __name__=="__main__":
    # build the environment
    env_config = {"action_type": "discrete"}
    env_class=EnvScheduling
    rl_name = "PPO"
    hyperparam_space_default_value, _ = rl_hyperparameter_space_dict_factory(rl_name)

    hyperparameters = hyperparam_space_default_value.copy()
    for k,v in hyperparam_space_default_value.items():
        hyperparameters[k]=v[len(v)//2]

    # update value similar to the publication
    hyperparameters["lr"]=3e-4
    hyperparameters["deep"]=2
    hyperparameters["wide"]=16
    hyperparameters["train_batch_size"]=64
    hyperparameters["sgd_minibatch_size"]=64
    hyperparameters["lambda"]=0.99
    rllib_trainer = rl_agent_factory(rl_name, hyperparameters, env_class, env_config=env_config)


    agent=AgentRLLIB(rllib_trainer, env_class, env_config, inv_prepro_state, inv_prepro_reward)
    episodes=10
    for i in range(episodes):
        for i in range(10):
            agent.train()

        score=agent.evaluate()
        print("Score:", score)


    # Basic
    #agent = AgentBestFit(None, env_class, env_config, inv_prepro_state, inv_prepro_reward)
    #score = agent.evaluate()
    #print("Score:", score)

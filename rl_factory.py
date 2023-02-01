#from ray.rllib.agents import dqn,a3c,ddpg,pg,ppo
from ray.rllib.algorithms import dqn,a3c,ddpg,pg,ppo
import copy

def _common_agent_config(default_config, env_config, hyperparameters):
    rl_algo_config=copy.deepcopy(default_config)
    rl_algo_config["log_level"] = "WARN"
    #rl_algo_config["evaluation_interval"] = 0
    #rl_algo_config["evaluation_num_episodes"] = 0

    rl_algo_config["preprocessor_pref"] = "rllib"
    rl_algo_config["env_config"]=env_config

    #rl_algo_config["num_gpus"]=0
    #rl_algo_config["num_gpus_per_worker"]=0
    return rl_algo_config

def _common_env_config(env_config):
    pass
    #env_config["evaluation_duration"] = 1

def rl_agent_factory(name, hyperparameters, env, env_config={}):
    _common_env_config(env_config)

    name=name.upper()
    if name=="DQN":
        rl_algo_config=_common_agent_config(dqn.DEFAULT_CONFIG.copy(),env_config,hyperparameters)
        # HP
        hidden=[hyperparameters["wide"] for i in range(hyperparameters["deep"])]
        rl_algo_config["hiddens"] = hidden
        rl_algo_config["lr"] = hyperparameters["lr"]
        rl_algo_config["gamma"]=hyperparameters["gamma"]
        rl_algo_config["noisy"] = False
        rl_algo_config["train_batch_size"] = hyperparameters["train_batch_size"]
        rl_algo_config["exploration_fraction"] = hyperparameters["exploration_fraction"]
        rl_algo_config["exploration_final_eps"] = hyperparameters["exploration_final_eps"]
        rl_algo_config["dueling"]=(hyperparameters["type"] == "dueling")

        trainer = dqn.DQN(env=env, config=rl_algo_config)
    elif name=="A3C":
        rl_algo_config = _common_agent_config(a3c.DEFAULT_CONFIG.copy(), env_config,hyperparameters)

        rl_algo_config["env_config"] = env_config
        rl_algo_config["evaluation_config"] = {
            "env_config": rl_algo_config["env_config"]
        }

        # dqn parameters and hyperparameters
        rl_algo_config["lr"] = hyperparameters["lr"]
        rl_algo_config["train_batch_size"] = hyperparameters["train_batch_size"]
        rl_algo_config["gamma"] = hyperparameters["gamma"]
        rl_algo_config["entropy_coeff"] = hyperparameters["entropy_coeff"]
        rl_algo_config["vf_loss_coeff"] = hyperparameters["vf_loss_coeff"]
        rl_algo_config["lambda"] = hyperparameters["lambda"]

        #https://github.com/ray-project/ray/blob/master/rllib/models/catalog.py
        rl_algo_config["model"]["fcnet_activation"]=hyperparameters["activation"]
        rl_algo_config["model"]["fcnet_hiddens"]=[hyperparameters["wide"] for i in range(hyperparameters["deep"])]
        rl_algo_config["model"]["use_lstm"] = False #config["use_lstm"] #use a 256 units LSTM


        trainer = a3c.A3CTrainer(env=env, config=rl_algo_config)
    elif name == "DDPG":#WARNING FOR CONTINUE CONTINUE-ACTION ONLY
        rl_algo_config = _common_agent_config(ddpg.DEFAULT_CONFIG.copy(), env_config,hyperparameters)

        nn=[hyperparameters["wide"] for i in range(hyperparameters["deep"])]
        rl_algo_config["actor_hiddens"]=nn
        rl_algo_config["critic_hiddens"]=nn
        rl_algo_config["actor_lr"]=hyperparameters["actor_lr"]
        rl_algo_config["critic_lr"]=hyperparameters["critic_lr"]
        rl_algo_config["tau"]=hyperparameters["tau"]
        rl_algo_config["gamma"]=hyperparameters["gamma"]
        rl_algo_config["train_batch_size"]=hyperparameters["train_batch_size"]

        trainer = ddpg.DDPG(env=env, config=rl_algo_config)
    elif name=="PG":
        rl_algo_config = _common_agent_config(pg.DEFAULT_CONFIG.copy(), env_config, hyperparameters)

        nn=[hyperparameters["wide"] for i in range(hyperparameters["deep"])]
        rl_algo_config["lr"]=hyperparameters["lr"]
        rl_algo_config["model"]["fcnet_hiddens"]=nn
        rl_algo_config["gamma"]=hyperparameters["gamma"]
        rl_algo_config["train_batch_size"]=hyperparameters["train_batch_size"]

        trainer = pg.PG(env=env, config=rl_algo_config)
    elif name=="PPO":
        rl_algo_config = _common_agent_config(ppo.DEFAULT_CONFIG.copy(), env_config, hyperparameters)

        nn=[hyperparameters["wide"] for i in range(hyperparameters["deep"])]
        rl_algo_config["lr"]=hyperparameters["lr"]
        rl_algo_config["model"]["fcnet_hiddens"]=nn
        #rl_algo_config["gamma"]=hyperparameters["gamma"]
        rl_algo_config["train_batch_size"]=hyperparameters["train_batch_size"]
        rl_algo_config["sgd_minibatch_size"]=hyperparameters["sgd_minibatch_size"]

        trainer = ppo.PPO(env=env, config=rl_algo_config)
    else:
        raise ValueError(f"ERROR rl_agent_factory not understood {name}")

    return trainer


def rl_hyperparameter_space_dict_factory(name):
    """
    Build hyperparameter space containing a list of potential values.
    It returns also which hyperparameters are mutable at training time. This concept of mutable parameters have been
     introduced in Population Based Training (PBT) paper. Most of hyperparameter optimizers does not mute
     hyperparameters.
    :param name: RL algo name: "PPO", "DQN", "DDPG", "PG", "A3C"
    :return: Tuple[ hyperparameter_space:dict[str,List] , mutable_variable_list:List[str]
    """
    lr_pv=[0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001]
    wide_pv= [32, 64, 128, 256, 512, 1024]
    deep_pv= [2, 3, 4, 5, 6, 7, 8]
    batch_size_pv=[16, 32, 64, 128, 256]
    sgd_minibatch=[16]
    gamma_pv=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
    exploration_final=[0., 0.01, 0.03, 0.05]
    exploration_fraction=[0.063, 0.125, 0.250, 0.500, 0.999]#1/16, 1/8, 1/6, 1/4, 1/2, 1/1

    name=name.upper()
    if name=="DQN":
        possible_values = {
            "lr": lr_pv,
            "wide": wide_pv,
            "deep": deep_pv,
            "type": ["dueling", "double_q"],
            "train_batch_size": batch_size_pv,
            "exploration_final_eps": exploration_final,
            "exploration_fraction": exploration_fraction,
            "gamma":gamma_pv,
            "action_type":["discrete"]}
        mutation_variables = ["lr", "train_batch_size", "exploration_fraction"]
    elif name=="A3C":
        possible_values = {
            "lr": lr_pv,
            "wide": wide_pv,
            "deep": deep_pv,#deep 6 leads to error
            "train_batch_size": batch_size_pv,
            "entropy_coeff": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
            "vf_loss_coeff": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            "lambda": gamma_pv,  # GAE lambda (term, part of the solution)
            "gamma": gamma_pv,  # GAE gamma (term, part of the problem)
            "activation":["relu","tanh","elu"],
            "action_type": ["discrete"] # discrete works less good than DQN. Check if mytrainable.trainer_prediction is ok
        }
        mutation_variables = ["lr", "train_batch_size", "entropy_coeff", "vf_loss_coeff", "lambda", "gamma"]
        # https://arxiv.org/pdf/1711.09846.pdf
        # We allow PBT to optimise the learning rate, entropy cost, and unroll length for UNREAL on DeepMind Lab,
        # learning rate, entropy cost, and intrinsic reward cost for FuN on Atari, and learning rate only
        # for A3C on StarCraft II
    elif name=="DDPG":
        possible_values = {
            "actor_lr": lr_pv,
            "critic_lr": lr_pv,
            "tau": [0.01, 0.003, 0.001, 0.0003],#"""lr""" of the target
            "wide": wide_pv,  # actor and critique
            "deep": deep_pv,
            "exploration_fraction": exploration_fraction,
            "exploration_final_scale":exploration_final,
            "train_batch_size": batch_size_pv,
            "gamma": gamma_pv,  # GAE gamma (term, part of the problem)
            "action_type": ["continuous"]
        }
        mutation_variables = ["actor_lr", "critic_lr", "train_batch_size", "gamma", "tau"]
    elif name=="PG":
        possible_values = {
            "lr": lr_pv,
            "wide": wide_pv,  # actor and critique
            "deep": deep_pv,
            "train_batch_size": batch_size_pv,
            "gamma": gamma_pv,  # GAE gamma (term, part of the problem)
            "action_type":["discrete"]
        }
        mutation_variables = ["lr", "train_batch_size", "gamma"]
    elif name=="PPO": #TODO ne fonctionne pas: ValueError: Trying to share variable default_policy/lr, but specified dtype float32 and found dtype float64_ref.
        possible_values = {
            "lr": lr_pv,
            "wide": wide_pv,  # actor and critique
            "deep": deep_pv,
            "train_batch_size": batch_size_pv,
            "sgd_minibatch_size": sgd_minibatch,
            "lambda": gamma_pv,  # GAE gamma (term, part of the problem)
            "action_type":["continuous"]
        }
        mutation_variables = ["lr", "train_batch_size", "gamma"]
    else:
        raise ValueError("ERROR rl_hyperparameter_space_dict_factory factor {name}")

    return possible_values, mutation_variables

def default_hyperparam_factory(name):
    """
    Default hyperparameters inspired of numerous research what are typical values
    :param name: RL algo name: "PPO", "DQN", "DDPG", "PG", "A3C"
    :return: dict[str,List] eg. {"lr":0.001, "batch_size": 64, ...}
    """
    space, _ = rl_hyperparameter_space_dict_factory(name)
    default_hp = dict()
    for k, v in space.items():
        default_hp[k] = v[len(v) // 2]  # common values are at the center
    return default_hp
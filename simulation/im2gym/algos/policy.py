from models.policy import DiagGaussianPd, RunningMeanStd, Policy, Value
from torch.distributions import Categorical, Uniform
import torch.nn as nn
import torch

class Model:
    def __init__(self, cfg, env_cfg):
        train_config = cfg["train"]["params"]["config"]
        normalize_input = train_config["normalize_input"]
        normalize_value = train_config["normalize_value"]
        mlp_config = cfg["train"]["params"]["network"]["mlp"]
        obs_shape = env_cfg["num_obs"]
        action_shape = env_cfg["num_acts"]
        self.asymmetric_obs=cfg["use_states"]
        self.policy = Policy(obs_shape, action_shape, mlp_config["units"], normalize_input, normalize_value, self.asymmetric_obs)
        if(self.asymmetric_obs):
            state_shape = env_cfg["num_states"]
            self.value_network=Value(state_shape, mlp_config["units"], normalize_input, normalize_value)
            self.value_network.to(cfg["rl_device"])
        self.policy.to(cfg["rl_device"])
        self.dist = DiagGaussianPd()
        
    def step(self, observation, state=None):
        with torch.no_grad():
            if self.asymmetric_obs:
                mu, logstd = self.policy(observation)
                value = self.value_network(state)
            else:
                mu, logstd, value = self.policy(observation)
            action = self.dist.sample(mu, logstd)
            neglogp = self.dist.neglogp(mu, logstd, action)
        return action, value, neglogp, mu, logstd

    def value(self, observation, state=None):
        with torch.no_grad():
            if self.asymmetric_obs:
                value = self.value_network(state)
            else:
                _, _, value = self.policy(observation)
        return value



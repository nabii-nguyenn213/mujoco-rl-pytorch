import os 
import torch 
import numpy as np 
from omegaconf import OmegaConf 
import gymnasium as gym 
from gymnasium.spaces import Box, Discrete

def loadConfig(configDir=None): 
    if configDir is None:
        configDir="configs/SAC.yaml"
    config = OmegaConf.load(configDir)
    OmegaConf.resolve(config)
    return config

def getObsActDim(env_name, **kwargs):
    env = gym.make(env_name, **kwargs)
    obs_space = env.observation_space
    act_space = env.action_space

    if isinstance(obs_space, Box):
        obs_dim = obs_space.shape[0]
    elif isinstance(obs_space, Discrete):
        obs_dim = obs_space.n
    else:
        raise NotImplementedError(f"Unsupported observation space: {type(obs_space)}")

    if isinstance(act_space, Box):
        act_dim = act_space.shape[0]
    elif isinstance(act_space, Discrete):
        act_dim = act_space.n
    else:
        raise NotImplementedError(f"Unsupported action space: {type(act_space)}")
    env.close()
    return obs_dim, act_dim

def getActionLimit(env_name, **kwargs): 
    env = gym.make(env_name, **kwargs)
    try: 
        act_space = env.action_space
        if not isinstance(act_space, gym.spaces.Box): 
            raise ValueError("Only continuous Box action spaces are supported.")
        if not np.allclose(act_space.high, act_space.high[0]):
            raise ValueError("Action bounds are not uniform across dimensions.")
        return float(act_space.high[0])
    finally:
        env.close()

def get_device(): 
    return "cuda" if torch.cuda.is_available() else "cpu"

def ensure_dir(path): 
    os.makedirs(path, exist_ok=True)

def dir_exist(path): 
    return os.path.isdir(path)

def file_exist(path): 
    return os.path.isfile(path) 

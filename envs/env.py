import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from envs.wrapper import ReturnWrapper

def make_env(env_name, max_episode_steps=0, reward_scaler=1.0, **env_kwargs):
    env = gym.make(env_name, **env_kwargs)
    if max_episode_steps and max_episode_steps > 0:
        env = TimeLimit(env, max_episode_steps)
    env = ReturnWrapper(env, reward_scaler=reward_scaler)
    return env

def make_env_demo(env_name, max_episode_steps=0, **env_kwargs):
    env = gym.make(env_name, render_mode="human", **env_kwargs)
    if max_episode_steps and max_episode_steps > 0:
        env = TimeLimit(env, max_episode_steps)
    env = ReturnWrapper(env)
    return env

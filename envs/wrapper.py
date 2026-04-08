import gymnasium as gym 

class ReturnWrapper(gym.Wrapper):
    def __init__(self, env, reward_scaler = 1.0):
        super().__init__(env)
        self.total_reward = 0.0
        self.reward_scaler = reward_scaler

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.total_reward += reward
        if terminated or truncated:
            info["episodic_return"] = self.total_reward
            self.total_reward = 0.0
        else:
            info["episodic_return"] = None
        return obs, reward * self.reward_scaler, terminated, truncated, info

    def reset(self, **kwargs):
        self.total_reward = 0
        return self.env.reset(**kwargs)

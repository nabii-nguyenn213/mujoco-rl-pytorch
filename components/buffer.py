import torch 
import numpy as np 

class ReplayBuffer: 
    def __init__(self, max_size, obs_dim, act_dim): 
        if isinstance(obs_dim, int):
            obs_dim = (obs_dim,)
        else:
            obs_dim = tuple(obs_dim)
        self.memory_size = int(max_size) 
        self.memory_counter = 0 
        self.state_memory = np.zeros((self.memory_size, *obs_dim), dtype=np.float32)
        self.next_state_memory = np.zeros((self.memory_size, *obs_dim), dtype=np.float32)
        self.action_memory = np.zeros((self.memory_size, act_dim), dtype=np.float32)
        self.reward_memory = np.zeros((self.memory_size, 1), dtype=np.float32)
        self.terminal_memory = np.zeros((self.memory_size, 1), dtype=np.float32)

    def store_transition(self, state, action, reward, next_state, done): 
        index = self.memory_counter % self.memory_size
        self.state_memory[index] = np.asarray(state, dtype=np.float32)
        self.next_state_memory[index] = np.asarray(next_state, dtype=np.float32)
        self.action_memory[index] = np.asarray(action, dtype=np.float32)
        self.reward_memory[index] = np.asarray(reward, dtype=np.float32)
        self.terminal_memory[index] = float(done)
        self.memory_counter += 1

    def __len__(self): 
        return min(self.memory_counter, self.memory_size)

    def can_sample(self, batch_size):
        return len(self) >= batch_size

    def sample_buffer(self, batch_size, device="cpu"): 
        max_mem=len(self) 
        if max_mem < batch_size: 
            raise ValueError(f"Not enough samples in replay buffer {max_mem} < {batch_size}")
        batch = np.random.randint(0, max_mem, size=batch_size) 
        return {
            "obs": torch.as_tensor(self.state_memory[batch], dtype=torch.float32, device=device),
            "act": torch.as_tensor(self.action_memory[batch], dtype=torch.float32, device=device),
            "rew": torch.as_tensor(self.reward_memory[batch], dtype=torch.float32, device=device),
            "next_obs": torch.as_tensor(self.next_state_memory[batch], dtype=torch.float32, device=device),
            "done": torch.as_tensor(self.terminal_memory[batch], dtype=torch.float32, device=device),
            }

class RolloutBuffer:
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, gae_lambda=0.95):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.max_size = size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.reset()

    def reset(self):
        self.obs_buf = np.zeros((self.max_size, self.obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((self.max_size, self.act_dim), dtype=np.float32)
        self.logp_buf = np.zeros((self.max_size,), dtype=np.float32)
        self.rew_buf = np.zeros((self.max_size,), dtype=np.float32)
        self.val_buf = np.zeros((self.max_size,), dtype=np.float32)
        self.next_val_buf = np.zeros((self.max_size,), dtype=np.float32)
        self.terminated_buf = np.zeros((self.max_size,), dtype=np.float32)
        self.episode_end_buf = np.zeros((self.max_size,), dtype=np.float32)
        self.adv_buf = np.zeros((self.max_size,), dtype=np.float32)
        self.ret_buf = np.zeros((self.max_size,), dtype=np.float32)
        self.ptr = 0
        self.path_ready = False

    def __len__(self):
        return self.ptr

    def is_full(self):
        return self.ptr >= self.max_size

    def add(self, obs, act, logp, reward, value, next_value, terminated, episode_end):
        if self.ptr >= self.max_size:
            raise ValueError("RolloutBuffer is full. Call reset() before adding more data.")
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.logp_buf[self.ptr] = logp
        self.rew_buf[self.ptr] = reward
        self.val_buf[self.ptr] = value
        self.next_val_buf[self.ptr] = next_value
        self.terminated_buf[self.ptr] = float(terminated)
        self.episode_end_buf[self.ptr] = float(episode_end)
        self.ptr += 1
        self.path_ready = False

    def compute_returns_and_advantages(self):
        size = self.ptr
        gae = 0.0
        for t in reversed(range(size)):
            bootstrap_mask = 1.0 - self.terminated_buf[t]
            gae_continue_mask = 1.0 - self.episode_end_buf[t]
            delta = (self.rew_buf[t] + self.gamma * bootstrap_mask * self.next_val_buf[t] - self.val_buf[t])
            gae = (delta + self.gamma * self.gae_lambda * gae_continue_mask * gae)
            self.adv_buf[t] = gae
        self.ret_buf[:size] = self.adv_buf[:size] + self.val_buf[:size]
        self.path_ready = True

    def get(self):
        if not self.path_ready:
            self.compute_returns_and_advantages()
        size = self.ptr
        rollout = {
            "obs": torch.tensor(self.obs_buf[:size], dtype=torch.float32),
            "act": torch.tensor(self.act_buf[:size], dtype=torch.float32),
            "logp": torch.tensor(self.logp_buf[:size], dtype=torch.float32),
            "adv": torch.tensor(self.adv_buf[:size], dtype=torch.float32),
            "ret": torch.tensor(self.ret_buf[:size], dtype=torch.float32),
            "val": torch.tensor(self.val_buf[:size], dtype=torch.float32),
        }
        return rollout

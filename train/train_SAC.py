import os
import numpy as np
import torch
from train.train_base import TrainAgent
from agents.SAC import SAC_Agent
from components.buffer import ReplayBuffer
from envs.env import make_env

class SAC(TrainAgent):
    def __init__(self, config, rank=None):
        super().__init__(config, rank=rank)
        max_episode_steps = self._get("env.max_episode_steps", 0)
        self.env_kwargs = self._get("env.kwargs", {}) or {} 
        self.env = make_env(self.env_id, max_episode_steps=max_episode_steps, **self.env_kwargs)
        self.obs, _ = self.env.reset(seed=self.seed)
        self.env.action_space.seed(self.seed)
        assert len(self.env.observation_space.shape) == 1,"ClassicalSACExperiment currently expects vector observations."
        assert len(self.env.action_space.shape) == 1,"ClassicalSACExperiment currently expects continuous 1D Box action space."
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]
        self.agent = SAC_Agent(self.config)
        self.replay_buffer = ReplayBuffer(
            max_size=self.memory_size,
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
        )
        self.last_log_info = None

    def _random_policy_action(self):
        return np.random.uniform(
            low=-1.0,
            high=1.0,
            size=(self.act_dim,)
        ).astype(np.float32)

    def _policy_to_env_action(self, policy_action):
        low = self.env.action_space.low
        high = self.env.action_space.high
        action_scale = 0.5 * (high - low)
        action_bias = 0.5 * (high + low)
        env_action = action_bias + action_scale * policy_action
        return np.clip(env_action, low, high)

    def _build_train_batch(self):
        batch_cpu = self.replay_buffer.sample_buffer(self.batch_size, device="cpu")
        obs_np = batch_cpu["obs"].numpy()
        next_obs_np = batch_cpu["next_obs"].numpy()
        batch = {
            "obs": torch.tensor(obs_np, dtype=torch.float32, device=self.agent.device),
            "act": batch_cpu["act"].to(self.agent.device),
            "rew": batch_cpu["rew"].to(self.agent.device),
            "next_obs": torch.tensor(next_obs_np, dtype=torch.float32, device=self.agent.device),
            "done": batch_cpu["done"].to(self.agent.device),
        }
        return batch

    @torch.no_grad()
    def evaluate(self):
        max_episode_steps = self._get("env.max_episode_steps", 0)
        eval_env = make_env(self.env_id, max_episode_steps=max_episode_steps, **self.env_kwargs)

        returns = []

        for ep in range(self.eval_episodes):
            obs, _ = eval_env.reset(seed=self.seed + 1000 + ep)
            done = False
            ep_return = 0.0

            while not done:
                policy_action = self.agent.act(obs, deterministic=True)
                env_action = self._policy_to_env_action(policy_action)

                obs, reward, terminated, truncated, info = eval_env.step(env_action)
                done = terminated or truncated
                ep_return += reward

            returns.append(ep_return)
        eval_env.close()
        return float(np.mean(returns))

    def run(self):
        self._setup_dirs()
        self._init_logger()
        self.logger.info(f"Initialized ClassicalSACExperiment on {self.env_id}")
        self.logger.info(f"Rank {self.rank} using seed {self.seed}")
        try:
            while self.global_step < self.total_timesteps:
                if self.global_step < self.learning_start:
                    policy_action = self._random_policy_action()
                else:
                    policy_action = self.agent.act(self.obs, deterministic=False)

                env_action = self._policy_to_env_action(policy_action)

                next_obs, reward, terminated, truncated, info = self.env.step(env_action)
                done = terminated or truncated

                self.replay_buffer.store_transition(
                    state=self.obs,
                    action=policy_action,
                    reward=reward,
                    next_state=next_obs,
                    done=done,
                )
                self.obs = next_obs
                self.episode_reward += reward
                self.episode_len += 1
                self.global_step += 1
                if self.replay_buffer.memory_counter >= max(self.batch_size, self.learning_start):
                    for _ in range(self.gradient_step):
                        batch = self._build_train_batch()
                        self.last_log_info = self.agent.update(batch)
                if self.global_step % self.log_every == 0 and self.last_log_info is not None:
                    self.logger.log_train(
                        step=self.global_step,
                        metrics=self.last_log_info,
                        print_to_console=True if self.rank is None or self.rank == 0 else False,
                    )
                if self.global_step > 0 and self.global_step % self.save_every == 0:
                    ckpt_path = os.path.join(self.ckpt_dir, f"sac_step_{self.global_step}.pt")
                    self._save(ckpt_path)
                    self.logger.log_checkpoint(ckpt_path, step=self.global_step, kind="checkpoint")
                if self.global_step > 0 and self.global_step % self.eval_every == 0:
                    avg_return = self.evaluate()
                    is_best = self.logger.log_eval(self.global_step, avg_return)
                    if is_best:
                        best_path = os.path.join(self.best_dir, "sac_best.pt")
                        self._save(best_path)
                        self.logger.log_checkpoint(best_path, step=self.global_step, kind="best")
                if done:
                    episodic_return = info.get("episodic_return", self.episode_reward)
                    self.episode_idx += 1
                    self.logger.log_episode(
                        episode=self.episode_idx,
                        step=self.global_step,
                        episodic_return=episodic_return,
                        episode_length=self.episode_len,
                    )
                    self.obs, _ = self.env.reset()
                    self.reset_episode_stats()
            final_path = os.path.join(self.model_dir, "sac_final.pt")
            self._save(final_path)
            self.logger.log_checkpoint(final_path, step=self.global_step, kind="final_model")
        finally:
            self.close()
            self.env.close()

    def _save(self, path):
            ckpt = {
                "global_step": self.global_step,
                "episode_idx": self.episode_idx,
                "config": self.config,
                "actor_critic_nets": self.agent.net.state_dict(),
                "target_critic1": self.agent.target_critic1.state_dict(),
                "target_critic2": self.agent.target_critic2.state_dict(),
                "actor_optim": self.agent.actor_optimizer.state_dict(),
                "critic1_optim": self.agent.crtic1_optimizer.state_dict(),
                "critic2_optim": self.agent.crtic2_optimizer.state_dict(),
            }
            torch.save(ckpt, path)

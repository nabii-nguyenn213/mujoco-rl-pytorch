import os
import numpy as np
import torch
from train.train_base import TrainAgent
from agents.PPO import PPO_Agent
from components.buffer import RolloutBuffer
from envs.env import make_env

class PPO(TrainAgent):
    def __init__(self, config, rank=None):
        super().__init__(config, rank=rank)
        max_episode_steps = self._get("env.max_episode_steps", 0)
        self.env_kwargs = self._get("env.kwargs", {}) or {}
        self.rollout_steps = int(self._get("train.rollout_steps", 2048))
        self.env = make_env(self.env_id, max_episode_steps=max_episode_steps, **self.env_kwargs)
        self.obs, _ = self.env.reset(seed=self.seed)
        self.env.action_space.seed(self.seed)
        assert len(self.env.observation_space.shape) == 1,"PPO currently expects vector observations."
        assert len(self.env.action_space.shape) == 1,"PPO currently expects continuous 1D Box action space."
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]
        self.agent = PPO_Agent(self.config)
        self.rollout_buffer = RolloutBuffer(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            size=self.rollout_steps,
            gamma=self.agent.gamma,
            gae_lambda=self.agent.gae_lambda,
        )
        self.last_log_info = None
        self.next_log_step = self.log_every
        self.next_eval_step = self.eval_every
        self.next_save_step = self.save_every

    def _policy_to_env_action(self, policy_action):
        low = self.env.action_space.low
        high = self.env.action_space.high
        action_scale = 0.5 * (high - low)
        action_bias = 0.5 * (high + low)
        env_action = action_bias + action_scale * policy_action
        return np.clip(env_action, low, high)

    @torch.no_grad()
    def evaluate(self):
        max_episode_steps = self._get("env.max_episode_steps", 0)
        eval_env = make_env(self.env_id,max_episode_steps=max_episode_steps,**self.env_kwargs)
        returns = []
        for ep in range(self.eval_episodes):
            obs, _ = eval_env.reset(seed=self.seed + 1000 + ep)
            done = False
            ep_return = 0.0
            while not done:
                policy_action, _, _ = self.agent.act(obs, deterministic=True)
                env_action = self._policy_to_env_action(policy_action)
                obs, reward, terminated, truncated, info = eval_env.step(env_action)
                done = terminated or truncated
                ep_return += reward
            returns.append(ep_return)
        eval_env.close()
        return float(np.mean(returns))

    def _collect_rollout(self):
        self.rollout_buffer.reset()
        while (not self.rollout_buffer.is_full()) and (self.global_step < self.total_timesteps):
            policy_action, logp, value = self.agent.act(self.obs, deterministic=False)
            env_action = self._policy_to_env_action(policy_action)
            next_obs, reward, terminated, truncated, info = self.env.step(env_action)
            episode_end = terminated or truncated
            if terminated:
                next_value = 0.0
            else:
                next_value = self.agent.get_value(next_obs)
            self.rollout_buffer.add(
                obs=self.obs,
                act=policy_action,
                logp=logp,
                reward=reward,
                value=value,
                next_value=next_value,
                terminated=terminated,
                episode_end=episode_end,
            )
            self.global_step += 1
            self.episode_reward += reward
            self.episode_len += 1
            if episode_end:
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
            else:
                self.obs = next_obs
        return self.rollout_buffer.get()

    def run(self):
        self._setup_dirs()
        self._init_logger()
        self.logger.info(f"Initialized PPO on {self.env_id}")
        self.logger.info(f"Rank {self.rank} using seed {self.seed}")
        try:
            while self.global_step < self.total_timesteps:
                rollout = self._collect_rollout()
                self.last_log_info = self.agent.update(rollout)
                if self.last_log_info is not None and self.global_step >= self.next_log_step:
                    self.logger.log_train(
                        step=self.global_step,
                        metrics=self.last_log_info,
                        print_to_console=True,
                    )
                    self.next_log_step = self.global_step + self.log_every
                if self.global_step >= self.next_save_step:
                    ckpt_path = os.path.join(self.ckpt_dir, f"ppo_step_{self.global_step}.pt")
                    self._save(ckpt_path)
                    self.logger.log_checkpoint(
                        ckpt_path,
                        step=self.global_step,
                        kind="checkpoint",
                    )
                    self.next_save_step = self.global_step + self.save_every
                if self.global_step >= self.next_eval_step:
                    avg_return = self.evaluate()
                    is_best = self.logger.log_eval(self.global_step, avg_return)
                    if is_best:
                        best_path = os.path.join(self.best_dir, "ppo_best.pt")
                        self._save(best_path)
                        self.logger.log_checkpoint(
                            best_path,
                            step=self.global_step,
                            kind="best",
                        )
                    self.next_eval_step = self.global_step + self.eval_every
            final_path = os.path.join(self.model_dir, "ppo_final.pt")
            self._save(final_path)
            self.logger.log_checkpoint(
                final_path,
                step=self.global_step,
                kind="final_model",
            )
        finally:
            self.close()
            if self.env is not None:
                self.env.close()

    def _save(self, path):
        ckpt = {
            "global_step": self.global_step,
            "episode_idx": self.episode_idx,
            "config": self.config,
            "actor_critic_nets": self.agent.net.state_dict(),
            "actor_optim": self.agent.actor_optimizer.state_dict(),
            "critic_optim": self.agent.critic_optimizer.state_dict(),
        }
        torch.save(ckpt, path)

    def save_model(self, save_path):
        self._save(save_path)

    def load_model(self, load_path):
        checkpoint = torch.load(load_path, map_location=self.agent.device, weights_only=False)
        self.global_step = int(checkpoint.get("global_step", 0))
        self.episode_idx = int(checkpoint.get("episode_idx", 0))
        self.agent.net.load_state_dict(checkpoint["actor_critic_nets"])
        self.agent.actor_optimizer.load_state_dict(checkpoint["actor_optim"])
        self.agent.critic_optimizer.load_state_dict(checkpoint["critic_optim"])

import torch 
import torch.nn as nn 
import torch.optim as optim 

from components.networks import ActorVCriticNetwork
from utils.helper import getObsActDim, get_device

EPS = 1e-6 
LOG_STD_MIN = -20 
LOG_STD_MAX = 2

class PPO_Agent: 
    def __init__(self, config): 
        self.config = config 
        self.device = config["train"]["device"]
        if self.device == "auto": 
            self.device = get_device()
        self.gamma = float(config["train"]["gamma"])
        self.gae_lambda = float(config["train"]["gae_lambda"])
        self.clip_coef = float(config["train"]["clip_coef"])
        self.ent_coef = float(config["train"]["ent_coef"])
        self.vf_coef = float(config["train"]["vf_coef"])
        self.max_grad_norm = float(config["train"]["max_grad_norm"])
        self.update_epochs = int(config["train"]["update_epochs"])
        self.minibatch_size = int(config["train"]["batch_size"])
        self.normalize_advantage = bool(config["train"].get("normalize_advantage", True))
        self.target_kl = config["train"].get("target_kl", None)
        self.target_kl = None if self.target_kl is None else float(self.target_kl)

        env_kwargs = config["env"].get("kwargs", {}) or {}
        obs_dim, act_dim = getObsActDim(config["env"]["name"], **env_kwargs)
        action_lim = config["env"]["action_lim"]
        hidden_size_actor = config["train"]["hidden_size_actor"]
        hidden_size_critic = config["train"]["hidden_size_critic"]
        hidden_act = config["train"].get("hidden_act", "Tanh")

        self.net = ActorVCriticNetwork(
            obs_dim=obs_dim,
            act_dim=act_dim,
            action_lim=action_lim,
            hidden_size_actor=hidden_size_actor,
            hidden_size_critic=hidden_size_critic,
            hidden_act=hidden_act,
        ).to(self.device)
        opt_name = config["train"]["optimizer"]["name"]
        actor_lr = float(config["train"]["optimizer"]["actor_lr"])
        critic_lr = float(config["train"]["optimizer"]["critic_lr"])
        if opt_name != "Adam":
            raise ValueError(f"Unsupported optimizer {opt_name}")
        self.actor_optimizer = optim.Adam(self.net.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.net.critic.parameters(), lr=critic_lr)

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        if not torch.is_tensor(obs):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        if deterministic:
            action = self.net.act_deterministic(obs)
            value = self.net.getValue(obs)
            log_prob = torch.zeros(obs.shape[0], device=self.device)
        else:
            action, log_prob, value, _ = self.net.sample_action(obs)
        return (
            action.squeeze(0).cpu().numpy(),
            log_prob.squeeze(0).cpu().item(),
            value.squeeze(0).cpu().item(),
        )

    @torch.no_grad()
    def get_value(self, obs):
        if not torch.is_tensor(obs):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        value = self.net.getValue(obs)
        return value.squeeze(0).cpu().item()

    def update(self, rollout):
        obs = rollout["obs"].to(self.device)
        act = rollout["act"].to(self.device)
        logp_old = rollout["logp"].to(self.device)
        ret = rollout["ret"].to(self.device)
        adv = rollout["adv"].to(self.device)
        if self.normalize_advantage:
            adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        n = obs.shape[0]
        minibatch_size = min(self.minibatch_size, n)
        metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clipfrac": 0.0,
            "total_loss": 0.0,
        }
        num_updates = 0
        early_stop = False
        for _ in range(self.update_epochs):
            indices = torch.randperm(n, device=self.device)
            for start in range(0, n, minibatch_size):
                mb_idx = indices[start:start + minibatch_size]
                mb_obs = obs[mb_idx]
                mb_act = act[mb_idx]
                mb_logp_old = logp_old[mb_idx]
                mb_ret = ret[mb_idx]
                mb_adv = adv[mb_idx]
                new_logp, entropy, value = self.net.evaluate_actions(mb_obs, mb_act)
                log_ratio = new_logp - mb_logp_old
                ratio = torch.exp(log_ratio)
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(
                    ratio,
                    1.0 - self.clip_coef,
                    1.0 + self.clip_coef,
                )
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()
                value_loss = 0.5 * ((value - mb_ret) ** 2).mean()
                entropy_bonus = entropy.mean()
                total_loss = (
                    policy_loss
                    + self.vf_coef * value_loss
                    - self.ent_coef * entropy_bonus
                )
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean().abs()
                    clipfrac = ((ratio - 1.0).abs() > self.clip_coef).float().mean()
                metrics["policy_loss"] += policy_loss.item()
                metrics["value_loss"] += value_loss.item()
                metrics["entropy"] += entropy_bonus.item()
                metrics["approx_kl"] += approx_kl.item()
                metrics["clipfrac"] += clipfrac.item()
                metrics["total_loss"] += total_loss.item()
                num_updates += 1
                if self.target_kl is not None and approx_kl.item() > self.target_kl:
                    early_stop = True
                    break

            if early_stop:
                break

        for k in metrics:
            metrics[k] /= max(num_updates, 1)

        with torch.no_grad():
            value_pred = self.net.getValue(obs)
            var_y = torch.var(ret)
            if var_y > 1e-8:
                explained_var = 1.0 - torch.var(ret - value_pred) / var_y
                explained_var = explained_var.item()
            else:
                explained_var = 0.0

        return {
            "critic_loss": metrics["value_loss"],
            "q1_loss": metrics["entropy"],
            "q2_loss": metrics["approx_kl"],
            "actor_loss": metrics["policy_loss"],
            "q1_mean": metrics["clipfrac"],
            "q2_mean": explained_var,
            "log_pi_mean": -metrics["entropy"],

            "ppo_policy_loss": metrics["policy_loss"],
            "ppo_value_loss": metrics["value_loss"],
            "ppo_entropy": metrics["entropy"],
            "ppo_approx_kl": metrics["approx_kl"],
            "ppo_clipfrac": metrics["clipfrac"],
            "ppo_explained_var": explained_var,
            "ppo_total_loss": metrics["total_loss"],
        }

    def save_model(self, path):
        torch.save(
            {
                "actor_critic_nets": self.net.state_dict(),
                "actor_optim": self.actor_optimizer.state_dict(),
                "critic_optim": self.critic_optimizer.state_dict(),
            },
            path,
        )

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.net.load_state_dict(checkpoint["actor_critic_nets"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optim"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optim"])

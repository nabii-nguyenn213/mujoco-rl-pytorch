import copy
import torch
import torch.nn.functional as F
import torch.optim as optim
from components.networks import ActorDoubleQCriticNetwork
from utils.helper import getObsActDim

class CSAC_Agent:
    def __init__(self, config):
        self.config = config

        self.device = config["train"]["device"]
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.gamma = float(config["train"]["gamma"])
        self.tau = float(config["train"]["tau"])  # target critic soft update
        self.sigma = float(config["train"]["sigma"])
        self.tau_rel = float(config["train"]["tau_rel"])

        env_kwargs = config["env"].get("kwargs", {}) or {}
        obs_dim, act_dim = getObsActDim(config["env"]["name"], **env_kwargs)

        hidden_size_actor = config["train"]["hidden_size_actor"]
        hidden_size_critic = config["train"]["hidden_size_critic"]
        actor_lr = float(config["train"]["optimizer"]["actor_lr"])
        critic_lr = float(config["train"]["optimizer"]["critic_lr"])

        self.net = ActorDoubleQCriticNetwork(
            obs_dim, act_dim, hidden_size_actor, hidden_size_critic
        ).to(self.device)

        # Target critics
        self.target_critic1 = copy.deepcopy(self.net.critic1).to(self.device)
        self.target_critic2 = copy.deepcopy(self.net.critic2).to(self.device)

        for p in self.target_critic1.parameters():
            p.requires_grad = False
        for p in self.target_critic2.parameters():
            p.requires_grad = False

        # Previous policy
        self.prev_actor = copy.deepcopy(self.net.actor).to(self.device)
        for p in self.prev_actor.parameters():
            p.requires_grad = False

        # Optimizers
        if config["train"]["optimizer"]["name"] == "Adam":
            self.actor_optimizer = optim.Adam(self.net.actor.parameters(), lr=actor_lr)
            self.critic1_optimizer = optim.Adam(self.net.critic1.parameters(), lr=critic_lr)
            self.critic2_optimizer = optim.Adam(self.net.critic2.parameters(), lr=critic_lr)
        else:
            raise ValueError(
                f"Unsupported optimizer {config['train']['optimizer']['name']}"
            )

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        if not torch.is_tensor(obs):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)

        if deterministic:
            action = self.net.act_deterministic(obs)
        else:
            action, _, _ = self.net.sample_action(obs)

        return action.squeeze(0).cpu().numpy()

    def update(self, batch):
        obs = batch["obs"].to(self.device)
        act = batch["act"].to(self.device)
        rew = batch["rew"].to(self.device)
        next_obs = batch["next_obs"].to(self.device)
        done = batch["done"].to(self.device).float()

        if rew.dim() == 1:
            rew = rew.unsqueeze(-1)
        if done.dim() == 1:
            done = done.unsqueeze(-1)

        # Critic update
        with torch.no_grad():
            next_action, next_log_pi, _ = self.net.sample_action(next_obs)

            target_q1_next = self.target_critic1(next_obs, next_action)
            target_q2_next = self.target_critic2(next_obs, next_action)
            target_q_next = torch.min(target_q1_next, target_q2_next)

            prev_next_log_pi = self.prev_actor.log_prob(next_obs, next_action)

            target = rew + self.gamma * (1.0 - done) * (
                target_q_next
                - self.sigma * next_log_pi
                - self.tau_rel * (next_log_pi - prev_next_log_pi)
            )

        current_q1 = self.net.critic1(obs, act)
        current_q2 = self.net.critic2(obs, act)

        q1_loss = F.mse_loss(current_q1, target)
        q2_loss = F.mse_loss(current_q2, target)
        critic_loss = q1_loss + q2_loss

        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        # Actor update
        new_action, log_pi, _ = self.net.sample_action(obs)

        q1_new = self.net.critic1(obs, new_action)
        q2_new = self.net.critic2(obs, new_action)
        q_new = torch.min(q1_new, q2_new)

        prev_log_pi = self.prev_actor.log_prob(obs, new_action)

        actor_loss = (
            (self.sigma + self.tau_rel) * log_pi
            - self.tau_rel * prev_log_pi
            - q_new
        ).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update previous actor AFTER actor step
        self.prev_actor.load_state_dict(self.net.actor.state_dict())

        # Soft update target critics
        self.soft_update(self.target_critic1, self.net.critic1)
        self.soft_update(self.target_critic2, self.net.critic2)

        return {
            "critic_loss": critic_loss.item(),
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "actor_loss": actor_loss.item(),
            "q1_mean": current_q1.mean().item(),
            "q2_mean": current_q2.mean().item(),
            "log_pi_mean": log_pi.mean().item(),
            "prev_log_pi_mean": prev_log_pi.mean().item(),
            "sigma": self.sigma,
            "tau_rel": self.tau_rel,
        }

    @torch.no_grad()
    def soft_update(self, target_net, source_net):
        for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1.0 - self.tau) * target_param.data
            )

    def save_model(self):
        data = {
            "actor_critic_nets": self.net.state_dict(),
            "target_critic1": self.target_critic1.state_dict(),
            "target_critic2": self.target_critic2.state_dict(),
            "prev_actor": self.prev_actor.state_dict(),
            "actor_optim": self.actor_optimizer.state_dict(),
            "critic1_optim": self.critic1_optimizer.state_dict(),
            "critic2_optim": self.critic2_optimizer.state_dict(),
        }

        torch.save(data, self.config["dir"]["model"])

    def load_model(self, path):
        model = torch.load(path, map_location=self.device, weights_only=False)

        self.net.load_state_dict(model["actor_critic_nets"])
        self.target_critic1.load_state_dict(model["target_critic1"])
        self.target_critic2.load_state_dict(model["target_critic2"])

        if "prev_actor" in model:
            self.prev_actor.load_state_dict(model["prev_actor"])
        else:
            # fallback for old checkpoints
            self.prev_actor.load_state_dict(self.net.actor.state_dict())

        self.actor_optimizer.load_state_dict(model["actor_optim"])
        self.critic1_optimizer.load_state_dict(model["critic1_optim"])
        self.critic2_optimizer.load_state_dict(model["critic2_optim"])

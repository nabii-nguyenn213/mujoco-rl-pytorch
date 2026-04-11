import copy 
import numpy as np
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
from components.networks import ActorDoubleQCriticNetwork
from utils.helper import getObsActDim

class SAC_Agent: 
    def __init__(self, config): 
        self.config = config 
        self.device = config["train"]["device"]
        if self.device == "auto": 
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gamma = config["train"]["gamma"]
        self.tau   = config["train"]["tau"]
        
        self.auto_alpha = config["train"].get("auto_alpha", False)
        self.fixed_alpha = float(config["train"].get("alpha", 0.2))
        self.alpha = torch.tensor(self.fixed_alpha, device=self.device)

        env_kwargs = config["env"].get("kwargs", {}) or {}
        obs_dim, act_dim = getObsActDim(config["env"]["name"], **env_kwargs)
    
        if self.auto_alpha: 
            target_entropy_cfg = config["train"].get("target_entropy", "auto") 
            if target_entropy_cfg == "auto": 
                self.target_entropy = -float(act_dim)
            else: 
                self.target_entropy = float(target_entropy_cfg) 
            init_alpha = float(config["train"].get("alpha", 0.2)) 
            self.log_alpha = torch.tensor([np.log(init_alpha)], dtype=torch.float32, device=self.device, requires_grad=True)
            alpha_lr = float(config["train"]["optimizer"].get("alpha_lr", "actor_lr"))
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr = alpha_lr) 
            self.alpha = self.log_alpha.exp().detach()
        else: 
            self.target_entropy = None 
            self.log_alpha = None 
            self.alpha_optimizer = None

        hidden_size_actor = config["train"]["hidden_size_actor"]
        hidden_size_critic = config["train"]["hidden_size_critic"]
        actor_lr = config["train"]["optimizer"]["actor_lr"]
        critic_lr = config["train"]["optimizer"]["critic_lr"]
        self.net = ActorDoubleQCriticNetwork(obs_dim, act_dim, hidden_size_actor, hidden_size_critic).to(self.device) 
        # Target critic 
        self.target_critic1 = copy.deepcopy(self.net.critic1).to(self.device)
        self.target_critic2 = copy.deepcopy(self.net.critic2).to(self.device)
        for p in self.target_critic1.parameters(): 
            p.requires_grad=False 
        for p in self.target_critic2.parameters(): 
            p.requires_grad=False 
        # Optimizer 
        if config["train"]["optimizer"]["name"] == "Adam": 
            self.actor_optimizer = optim.Adam(self.net.actor.parameters(), lr=actor_lr)
            self.crtic1_optimizer= optim.Adam(self.net.critic1.parameters(), lr=critic_lr)
            self.crtic2_optimizer= optim.Adam(self.net.critic2.parameters(), lr=critic_lr)
        else : 
            raise ValueError(f"Unsupported optimizer {config['train']['optimizer']['name']}")

    @torch.no_grad() 
    def act(self, obs, deterministic=False): 
        if not torch.is_tensor(obs): 
            obs=torch.tensor(obs, dtype=torch.float32, device=self.device)
        if deterministic: 
            action=self.net.act_deterministic(obs)
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
       
        with torch.no_grad():
            alpha = self.log_alpha.exp() if self.auto_alpha else self.alpha
            next_action, next_log_pi, _ = self.net.sample_action(next_obs)
            target_q1_next = self.target_critic1(next_obs, next_action)
            target_q2_next = self.target_critic2(next_obs, next_action)
            target_q_next = torch.min(target_q1_next, target_q2_next)
            target = rew + self.gamma * (1.0 - done) * (target_q_next - alpha * next_log_pi)

        current_q1 = self.net.critic1(obs, act) 
        current_q2 = self.net.critic2(obs, act) 
        
        q1_loss = F.mse_loss(current_q1, target)
        q2_loss = F.mse_loss(current_q2, target)
        critic_loss = q1_loss + q2_loss

        self.crtic1_optimizer.zero_grad()
        self.crtic2_optimizer.zero_grad()
        critic_loss.backward()
        self.crtic1_optimizer.step()
        self.crtic2_optimizer.step()

        new_action, log_pi, _ = self.net.sample_action(obs)
        q1_new = self.net.critic1(obs, new_action)
        q2_new = self.net.critic2(obs, new_action)
        q_new = torch.min(q1_new, q2_new)

        alpha = self.log_alpha.exp() if self.auto_alpha else self.alpha
        actor_loss = (alpha * log_pi - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp().detach()
        else:
            alpha_loss = torch.tensor(0.0, device=self.device)

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
            "alpha": float(self.alpha.item()),
            "alpha_loss": float(alpha_loss.item()),
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
            "actor_optim": self.actor_optimizer.state_dict(),
            "critic1_optim": self.crtic1_optimizer.state_dict(),
            "critic2_optim": self.crtic2_optimizer.state_dict(),
        }

        if self.auto_alpha:
            data["log_alpha"] = self.log_alpha.detach().cpu()
            data["alpha_optim"] = self.alpha_optimizer.state_dict()

        torch.save(data, self.config["dir"]["model"])

    def load_model(self, path):
        model = torch.load(path, map_location=self.device, weights_only=False)

        self.net.load_state_dict(model["actor_critic_nets"])
        self.target_critic1.load_state_dict(model["target_critic1"])
        self.target_critic2.load_state_dict(model["target_critic2"])

        self.actor_optimizer.load_state_dict(model["actor_optim"])
        self.crtic1_optimizer.load_state_dict(model["critic1_optim"])
        self.crtic2_optimizer.load_state_dict(model["critic2_optim"])

        if self.auto_alpha and "log_alpha" in model:
            self.log_alpha.data.copy_(model["log_alpha"].to(self.device))
            self.alpha_optimizer.load_state_dict(model["alpha_optim"])
            self.alpha = self.log_alpha.exp().detach()

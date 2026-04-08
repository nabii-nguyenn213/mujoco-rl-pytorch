import os 
import time
import random
import copy
import numpy as np
from datetime import datetime
import torch
from abc import ABC, abstractmethod
from utils.helper import loadConfig, ensure_dir, get_device
from utils.logger import Logger
try:
    from omegaconf import OmegaConf
except Exception:
    OmegaConf = None

class TrainAgent(ABC):
    def __init__(self, config, rank=None):
        if isinstance(config, str):
            self.config = loadConfig(config)
        else:
            self.config = config
        self.rank = rank
        self.device = self._get("train.device", "auto")
        if self.device == "auto":
            self.device = get_device()

        self.env_id = self._get("env.name")
        if self.env_id is None:
            raise ValueError("config['env']['name'] is required")

        self.total_timesteps = int(self._get("train.total_timesteps", 0))
        self.batch_size = int(self._get("train.batch_size", 128))
        self.memory_size = int(self._get("train.memory_size", 100000))
        self.learning_start = int(self._get("train.learning_start", 1000))
        self.gradient_step = int(self._get("train.gradient_step", 1))
        self.seed = int(self._get("train.seed", 42)) 
        if self.rank is not None: 
            self.seed += self.rank
        self.show_tb = bool(self._get("train.show_tb", False))
        self.run_name = self._get("train.run_name", self.__class__.__name__)

        self.eval_every = int(self._get("eval.eval_every", self._get("train.eval_every", 5000)))
        self.save_every = int(self._get("eval.save_every", self._get("train.save_every", 5000)))
        self.log_every = int(self._get("eval.log_every", self._get("train.log_every", 1000)))
        self.eval_episodes = int(self._get("eval.eval_episodes", self._get("train.eval_episodes", 5)))

        self.normalize_obs = bool(self._get("normalizer.normalize_obs", self._get("train.normalize_obs", False)))
        self.obs_norm_clip = float(self._get("normalizer.obs_norm_clip", self._get("train.obs_norm_clip", 10.0)))
        self.obs_norm_epsilon = float(self._get("normalizer.obs_norm_epsilon", self._get("train.obs_norm_epsilon", 1e-8)))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(self._get("dir.log"), f"{self.run_name}_{timestamp}")
        self.ckpt_dir = os.path.join(self._get("dir.ckpt"), f"{self.run_name}_{timestamp}") 
        self.model_dir = os.path.join(self._get("dir.model"), f"{self.run_name}_{timestamp}")
        self.best_dir = os.path.join(self._get("dir.best"), f"{self.run_name}_{timestamp}") 
        self.tb_dir = os.path.join(self._get("dir.tensorboard"), f"{self.run_name}_{timestamp}")
        if self.rank is not None: 
            self.log_dir = os.path.join(self.log_dir, f"rank_{rank}")
            self.ckpt_dir = os.path.join(self.ckpt_dir, f"rank_{rank}") 
            self.model_dir = os.path.join(self.model_dir, f"rank_{rank}")
            self.best_dir = os.path.join(self.best_dir, f"rank_{rank}") 
            self.tb_dir = os.path.join(self.tb_dir, f"rank_{rank}")

        self.start_time = time.time()
        self.global_step = 0
        self.episode_idx = 0
        self.best_eval_return = -np.inf

        self.env = None
        self.agent = None
        self.replay_buffer = None
        self.obs_normalizer = None

        self.obs = None
        self.obs_dim = None
        self.act_dim = None

        self.episode_reward = 0.0
        self.episode_len = 0

        self._set_seed()
        self._build_normalizer()

        self.logger = None

    def _get(self, path, default=None):
        keys = path.split(".")
        cur = self.config
        try:
            for key in keys:
                cur = cur[key]
            return cur
        except Exception:
            return default

    def _setup_dirs(self):
        for path in [self.log_dir, self.ckpt_dir, self.model_dir, self.best_dir, self.tb_dir]:
            if path is not None:
                ensure_dir(path)

    def _set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def _build_config_snapshot(self):
        if OmegaConf is not None:
            try:
                if OmegaConf.is_config(self.config):
                    cfg = OmegaConf.to_container(self.config, resolve=True)
                else:
                    cfg = copy.deepcopy(self.config)
            except Exception:
                cfg = copy.deepcopy(self.config)
        else:
            cfg = copy.deepcopy(self.config)

        cfg.setdefault("train", {})
        cfg["train"]["device"] = self.device
        cfg["train"]["seed"] = self.seed

        if self.obs_dim is not None:
            cfg["obs_dim"] = int(self.obs_dim)
        if self.act_dim is not None:
            cfg["act_dim"] = int(self.act_dim)

        return cfg

    def _init_logger(self):
        self.logger = Logger(
            self.config,
            logdir=self.log_dir, 
            tb_dir=self.tb_dir, 
            run_name=self.run_name,
            config_to_save=self._build_config_snapshot(),
        )

    def refresh_logger_config(self):
        if self.logger is not None:
            self.logger.update_config_to_save(self._build_config_snapshot())

    def reset_episode_stats(self):
        self.episode_reward = 0.0
        self.episode_len = 0

    def save_normalizer(self, path):
        if self.obs_normalizer is None:
            return
        torch.save(self.obs_normalizer.state_dict(), path)

    def close(self):
        if self.logger is not None:
            self.logger.close()

    @abstractmethod
    def evaluate(self):
        raise NotImplementedError("evaluate() must be implemented in subclass")

    @abstractmethod
    def run(self):
        raise NotImplementedError("run() must be implemented in subclass")

    def save_model(self, save_path): 
        pass

    def load_model(self, load_path): 
        pass 

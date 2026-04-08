import os 
import csv 
import time 
import copy 
from datetime import datetime 
from torch.utils.tensorboard import SummaryWriter
from utils.helper import ensure_dir
try : 
    import yaml 
except Exception: 
    yaml = None
try : 
    from omegaconf import OmegaConf
except Exception: 
    OmegaConf = None

class Logger: 
    def __init__(self, config, logdir=None, tb_dir=None, run_name=None, config_to_save=None): 
        self.config = config 
        self.config_to_save = self._freeze_config(config if config_to_save is None else config_to_save)

        self.run_name = run_name if run_name else "SAC" 
        self.start_time = time.time() 
        self.best_eval_return = float("-inf") 

        self.log_dir = str(config["dir"]["log"]) if logdir is None else logdir
        self.tb_dir = str(config["dir"]["tensorboard"]) if tb_dir is None else tb_dir
        self.show_tb = bool(config["train"]["show_tb"]) if "show_tb" in config["train"] else False

        ensure_dir(self.log_dir)
        ensure_dir(self.tb_dir)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{run_name}_{timestamp}"
        self.text_log_path = os.path.join(self.log_dir, f"{self.run_id}.log")
        self.train_csv_path = os.path.join(self.log_dir, f"{self.run_id}_train.csv")
        self.episode_csv_path = os.path.join(self.log_dir, f"{self.run_id}_episode.csv")
        self.eval_csv_path = os.path.join(self.log_dir, f"{self.run_id}_eval.csv")
        self.config_save_path = os.path.join(self.log_dir, f"{self.run_id}_config.yaml")

        self.writer = None
        if self.show_tb and SummaryWriter is not None:
            self.writer = SummaryWriter(log_dir=self.tb_dir)

        self._init_files()
        self.save_config() 
        self.info(f"Logger initialized: {self.run_id}")

        if self.writer is None and self.show_tb: 
            self.info("TensorBoard requested but unavailable. Install tensorboard to enable it.")

    def _init_files(self): 
        if not os.path.exists(self.train_csv_path): 
            with open(self.train_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["step","critic_loss","q1_loss","q2_loss","actor_loss","q1_mean","q2_mean","log_pi_mean"])
        if not os.path.exists(self.episode_csv_path):
            with open(self.episode_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["episode","step","episodic_return","episode_length","elapsed_sec"])
        if not os.path.exists(self.eval_csv_path):
            with open(self.eval_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["step","avg_return","best_so_far","elapsed_sec"])
        if not os.path.exists(self.text_log_path):
            with open(self.text_log_path, "w", encoding="utf-8") as f:
                f.write("")

    def _freeze_config(self, config): 
        if OmegaConf is not None:
            try:
                if OmegaConf.is_config(config):
                    return OmegaConf.to_container(config, resolve=True)
            except Exception:
                pass
        return copy.deepcopy(config)

    def update_config_to_save(self, config_to_save):
        self.config_to_save = self._freeze_config(config_to_save)
        self.save_config()

    def save_config(self): 
        try: 
            if yaml is not None: 
                with open(self.config_save_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(self.config_to_save, f, sort_keys=False)
            else:
                with open(self.config_save_path, "w", encoding="utf-8") as f:
                    f.write(str(self.config_to_save))
        except Exception as e:
            self.info(f"Failed to save config: {e}")

    def _elapsed_sec(self):
        return time.time() - self.start_time

    def _append_text(self, msg):
        with open(self.text_log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    def _to_float(self, x):
        try:
            return float(x)
        except Exception:
            return x
    def info(self, msg):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_msg = f"[{timestamp}] {msg}"
        print(full_msg)
        self._append_text(full_msg)

    def log_train(self, step, metrics: dict, print_to_console=False):
        critic_loss = self._to_float(metrics.get("critic_loss", 0.0))
        q1_loss = self._to_float(metrics.get("q1_loss", 0.0))
        q2_loss = self._to_float(metrics.get("q2_loss", 0.0))
        actor_loss = self._to_float(metrics.get("actor_loss", 0.0))
        q1_mean = self._to_float(metrics.get("q1_mean", 0.0))
        q2_mean = self._to_float(metrics.get("q2_mean", 0.0))
        log_pi_mean = self._to_float(metrics.get("log_pi_mean", 0.0))
        with open(self.train_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                step,
                critic_loss,
                q1_loss,
                q2_loss,
                actor_loss,
                q1_mean,
                q2_mean,
                log_pi_mean,
            ])
        if self.writer is not None:
            self.writer.add_scalar("train/critic_loss", critic_loss, step)
            self.writer.add_scalar("train/q1_loss", q1_loss, step)
            self.writer.add_scalar("train/q2_loss", q2_loss, step)
            self.writer.add_scalar("train/actor_loss", actor_loss, step)
            self.writer.add_scalar("train/q1_mean", q1_mean, step)
            self.writer.add_scalar("train/q2_mean", q2_mean, step)
            self.writer.add_scalar("train/log_pi_mean", log_pi_mean, step)
        if print_to_console:
            self.info(
                f"step={step} "
                f"critic_loss={critic_loss:.4f} "
                f"actor_loss={actor_loss:.4f} "
                f"log_pi={log_pi_mean:.4f} "
                f"q1={q1_mean:.4f} "
                f"q2={q2_mean:.4f}"
            )

    def log_episode(self, episode, step, episodic_return, episode_length):
        elapsed = self._elapsed_sec()
        with open(self.episode_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                episode,
                step,
                self._to_float(episodic_return),
                self._to_float(episode_length),
                elapsed,
            ])
        if self.writer is not None:
            self.writer.add_scalar("episode/return", self._to_float(episodic_return), step)
            self.writer.add_scalar("episode/length", self._to_float(episode_length), step)
        self.info(
            f"episode={episode} "
            f"step={step} "
            f"return={float(episodic_return):.3f} "
            f"length={int(episode_length)}"
        )

    def log_eval(self, step, avg_return):
        avg_return = self._to_float(avg_return)
        is_best = avg_return > self.best_eval_return
        if is_best:
            self.best_eval_return = avg_return
        elapsed = self._elapsed_sec()
        with open(self.eval_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                step,
                avg_return,
                self.best_eval_return,
                elapsed,
            ])

        if self.writer is not None:
            self.writer.add_scalar("eval/avg_return", avg_return, step)
            self.writer.add_scalar("eval/best_return", self.best_eval_return, step)
        if is_best:
            self.info(f"[eval] step={step} avg_return={avg_return:.3f} NEW_BEST")
        else:
            self.info(f"[eval] step={step} avg_return={avg_return:.3f}")
        return is_best

    def log_checkpoint(self, path, step=None, kind="checkpoint"):
        if step is None:
            self.info(f"saved {kind}: {path}")
        else:
            self.info(f"saved {kind} at step={step}: {path}")

    def close(self):
        self.info("Closing logger")
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None

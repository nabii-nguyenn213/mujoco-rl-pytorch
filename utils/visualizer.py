import argparse
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.env import make_env_demo
from agents.SAC import SAC_Agent
from utils.helper import loadConfig, dir_exist, file_exist


def getParser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Hopper-v5")
    parser.add_argument("--runid", type=str, default=None)
    parser.add_argument("--loadOption", type=str, default="best")
    parser.add_argument("--rank", type=int, default=None)
    return parser.parse_args()


def _get_base_root_from_load_option(env_name: str, load_option: str) -> Path:
    results_root = PROJECT_ROOT / "results"

    if load_option == "best":
        return results_root / "best" / env_name
    elif load_option == "final":
        return results_root / "models" / env_name
    elif load_option.startswith("checkpoint_"):
        # your path uses "checkpoints"
        return results_root / "checkpoints" / env_name
    else:
        raise ValueError("Only support `best`, `final` and `checkpoint_<step>`")


def _get_log_root(env_name: str) -> Path:
    """
    Expected log root:
        logs/log/<env_name>/

    Change this only if your log folder layout changes.
    """
    return PROJECT_ROOT / "logs" / "log" / env_name


def _resolve_run_dir(base_root: Path, run_id: str, rank=None) -> Path:
    if not dir_exist(str(base_root)):
        raise FileNotFoundError(f"Base directory not found: {base_root}")

    candidates = [base_root / run_id]
    candidates.extend(sorted(base_root.glob(f"*{run_id}*")))

    existing = []
    seen = set()

    for path in candidates:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        if dir_exist(str(path)):
            existing.append(path)

    if not existing:
        raise FileNotFoundError(
            f"Could not find a run directory under:\n{base_root}\n\n"
            f"Expected something like:\n"
            f"  {run_id}"
        )

    if len(existing) > 1:
        matches = "\n".join(str(p) for p in existing)
        raise FileExistsError(
            "Multiple run directories matched the provided --runid. "
            "Please pass a more specific run id or full folder name:\n"
            f"{matches}"
        )

    run_dir = existing[0]

    if rank is not None:
        rank_dir = run_dir / f"rank_{rank}"
        if not dir_exist(str(rank_dir)):
            raise FileNotFoundError(
                f"Rank directory not found: {rank_dir}\n"
                f"Make sure this run was trained with MPI and that --rank {rank} is correct."
            )
        return rank_dir

    return run_dir


def _resolve_checkpoint_path(run_dir: Path, load_option: str) -> Path:
    if load_option == "best":
        model_path = run_dir / "sac_best.pt"
    elif load_option == "final":
        model_path = run_dir / "sac_final.pt"
    elif load_option.startswith("checkpoint_"):
        step = load_option.split("_", 1)[1].strip()
        if not step.isdigit():
            raise ValueError(
                "Checkpoint format must be checkpoint_<step>, for example checkpoint_100000"
            )
        model_path = run_dir / f"sac_step_{step}.pt"
    else:
        raise ValueError("Only support `best`, `final` and `checkpoint_<step>`")

    if not file_exist(str(model_path)):
        raise FileNotFoundError(f"Checkpoint file not found: {model_path}")

    return model_path


def _find_saved_run_config(env_name: str, run_id: str, rank=None):
    log_root = _get_log_root(env_name)
    if not dir_exist(str(log_root)):
        return None

    try:
        run_log_dir = _resolve_run_dir(log_root, run_id, rank=rank)
    except Exception:
        return None

    config_files = sorted(run_log_dir.glob("*_config.yaml"))
    if len(config_files) == 0:
        return None

    return config_files[0]


def _sanitize_env_kwargs(env_name: str, env_kwargs):
    env_kwargs = dict(env_kwargs or {})
    if len(env_kwargs) == 0:
        return {}

    try:
        env = gym.make(env_name, **env_kwargs)
        env.close()
        return env_kwargs
    except Exception as e:
        print(f"[warning] Invalid env kwargs for {env_name}: {env_kwargs}")
        print(f"[warning] Falling back to empty kwargs. gym.make error: {e}")
        return {}


def _build_config(env_name: str, run_id: str, rank=None):
    saved_config_path = _find_saved_run_config(env_name, run_id, rank=rank)

    if saved_config_path is not None:
        config = loadConfig(str(saved_config_path))
    else:
        config_path = PROJECT_ROOT / "configs" / "SAC.yaml"
        config = loadConfig(str(config_path))

    config["dir"]["root"] = str(PROJECT_ROOT)
    config["env"]["name"] = env_name

    env_kwargs = config["env"].get("kwargs", {}) or {}
    config["env"]["kwargs"] = _sanitize_env_kwargs(env_name, env_kwargs)

    return config


def _build_agent(config):
    return SAC_Agent(config)


def _policy_to_env_action(env, policy_action):
    """
    Policy action is assumed to be normalized in [-1, 1].
    Map it once to the actual env action range.
    """
    if not hasattr(env.action_space, "low") or not hasattr(env.action_space, "high"):
        return policy_action

    policy_action = np.asarray(policy_action, dtype=np.float32)
    low = np.asarray(env.action_space.low, dtype=np.float32)
    high = np.asarray(env.action_space.high, dtype=np.float32)

    action_scale = 0.5 * (high - low)
    action_bias = 0.5 * (high + low)
    env_action = action_bias + action_scale * policy_action
    return np.clip(env_action, low, high)


def visualize(env_name: str, run_id: str, load_option: str, rank=None):
    config = _build_config(env_name, run_id, rank=rank)

    base_root = _get_base_root_from_load_option(env_name, load_option)
    run_dir = _resolve_run_dir(base_root, run_id, rank=rank)
    checkpoint_path = _resolve_checkpoint_path(run_dir, load_option)

    env_kwargs = config["env"].get("kwargs", {}) or {}
    max_episode_steps = int(config["env"].get("max_episode_steps", 0))
    num_episodes = int(config.get("eval", {}).get("eval_episodes", 3))
    seed = int(config["train"].get("seed", 42))

    if rank is not None:
        seed += rank

    agent = _build_agent(config)
    agent.load_model(str(checkpoint_path))

    env = make_env_demo(
        env_name,
        max_episode_steps=max_episode_steps,
        **env_kwargs,
    )

    print(f"Environment       : {env_name}")
    print(f"Run directory     : {run_dir}")
    print(f"Checkpoint loaded : {checkpoint_path}")
    print(f"Rank              : {rank}")
    print(f"Render mode       : human")
    print(f"Episodes          : {num_episodes}")

    try:
        for ep in range(num_episodes):
            obs, _ = env.reset(seed=seed + 1000 + ep)
            done = False
            ep_return = 0.0
            ep_len = 0

            try:
                env.render()
            except Exception:
                pass

            while not done:
                policy_action = agent.act(obs, deterministic=True)
                env_action = _policy_to_env_action(env=env, policy_action=policy_action)

                obs, reward, terminated, truncated, info = env.step(env_action)
                done = terminated or truncated
                ep_return += reward
                ep_len += 1

                try:
                    env.render()
                except Exception:
                    pass

            episodic_return = info.get("episodic_return", ep_return)
            print(
                f"episode={ep + 1}/{num_episodes} "
                f"return={episodic_return:.3f} length={ep_len}"
            )
    finally:
        env.close()


if __name__ == "__main__":
    args = getParser()

    env_name = args.env
    run_id = args.runid
    load_option = args.loadOption
    rank = args.rank

    if run_id is None:
        raise ValueError("'--runid' flag must be provided.")

    visualize(
        env_name=env_name,
        run_id=run_id,
        load_option=load_option,
        rank=rank,
    )

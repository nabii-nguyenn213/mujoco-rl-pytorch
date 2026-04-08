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
from agents.PPO import PPO_Agent
from utils.helper import loadConfig, dir_exist, file_exist


def getParser():
    """
    Example Usage

    # Load best agent
    python utils/visualizer.py --agent SAC --env Hopper-v5 --runid 20260408_163845 --loadOption best

    # Load final agent
    python utils/visualizer.py --agent SAC --env Hopper-v5 --runid 20260408_163845 --loadOption final

    # Load checkpoint agent
    python utils/visualizer.py --agent SAC --env Hopper-v5 --runid 20260408_163845 --loadOption checkpoint_100000

    # Load checkpoint agent for MPI rank 0
    python utils/visualizer.py --agent SAC --env Hopper-v5 --runid 20260408_163845 --loadOption checkpoint_100000 --rank 0
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, default="SAC")
    parser.add_argument("--env", type=str, default="Hopper-v5")
    parser.add_argument("--runid", type=str, default=None)
    parser.add_argument("--loadOption", type=str, default="best")
    parser.add_argument("--rank", type=int, default=None)
    return parser.parse_args()

def _normalize_agent_name(agent_name: str) -> str:
    agent_name = agent_name.strip().upper()
    if agent_name not in {"SAC", "PPO"}:
        raise ValueError(f"Unsupported agent '{agent_name}'. Only SAC and PPO are supported.")
    return agent_name


def _get_base_root_from_load_option(agent_name: str, env_name: str, load_option: str) -> Path:
    results_root = PROJECT_ROOT / "results"

    if load_option == "best":
        return results_root / "best" / agent_name / env_name
    elif load_option == "final":
        return results_root / "models" / agent_name / env_name
    elif load_option.startswith("checkpoint_"):
        return results_root / "checkpoints" / agent_name / env_name
    else:
        raise ValueError("Only support `best`, `final` and `checkpoint_<step>`")


def _get_log_root(agent_name: str, env_name: str) -> Path:
    return PROJECT_ROOT / "logs" / "log" / agent_name / env_name


def _resolve_run_dir(base_root: Path, agent_name: str, env_name: str, run_id: str, rank=None) -> Path:
    if not dir_exist(str(base_root)):
        raise FileNotFoundError(f"Base directory not found: {base_root}")

    candidates = [
        base_root / run_id,
        base_root / f"{agent_name}_{env_name}_{run_id}",
    ]
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
            f"  {agent_name}_{env_name}_{run_id}"
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


def _resolve_checkpoint_path(run_dir: Path, agent_name: str, load_option: str) -> Path:
    agent_prefix = agent_name.lower()

    if load_option == "best":
        model_path = run_dir / f"{agent_prefix}_best.pt"
    elif load_option == "final":
        model_path = run_dir / f"{agent_prefix}_final.pt"
    elif load_option.startswith("checkpoint_"):
        step = load_option.split("_", 1)[1].strip()
        if not step.isdigit():
            raise ValueError(
                "Checkpoint format must be checkpoint_<step>, for example checkpoint_100000"
            )
        model_path = run_dir / f"{agent_prefix}_step_{step}.pt"
    else:
        raise ValueError("Only support `best`, `final` and `checkpoint_<step>`")

    if not file_exist(str(model_path)):
        raise FileNotFoundError(f"Checkpoint file not found: {model_path}")

    return model_path


def _find_saved_run_config(agent_name: str, env_name: str, run_id: str, rank=None):
    log_root = _get_log_root(agent_name, env_name)
    if not dir_exist(str(log_root)):
        return None

    try:
        run_log_dir = _resolve_run_dir(log_root, agent_name, env_name, run_id, rank=rank)
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


def _build_config(agent_name: str, env_name: str, run_id: str, rank=None):
    saved_config_path = _find_saved_run_config(agent_name, env_name, run_id, rank=rank)

    if saved_config_path is not None:
        config = loadConfig(str(saved_config_path))
    else:
        config_path = PROJECT_ROOT / "configs" / f"{agent_name}.yaml"
        config = loadConfig(str(config_path))

    config["dir"]["root"] = str(PROJECT_ROOT)
    config["env"]["name"] = env_name

    env_kwargs = config["env"].get("kwargs", {}) or {}
    config["env"]["kwargs"] = _sanitize_env_kwargs(env_name, env_kwargs)

    return config


def _build_agent(agent_name: str, config):
    if agent_name == "SAC":
        return SAC_Agent(config)
    if agent_name == "PPO":
        return PPO_Agent(config)
    raise ValueError(f"Unsupported agent '{agent_name}'")


def _policy_to_env_action(env, policy_action, policy_action_lim=1.0):
    if not hasattr(env.action_space, "low") or not hasattr(env.action_space, "high"):
        return policy_action

    policy_action = np.asarray(policy_action, dtype=np.float32)
    low = np.asarray(env.action_space.low, dtype=np.float32)
    high = np.asarray(env.action_space.high, dtype=np.float32)

    policy_action_lim = float(policy_action_lim)
    if policy_action_lim <= 0:
        policy_action_lim = 1.0

    scaled = (policy_action + policy_action_lim) / (2.0 * policy_action_lim)
    env_action = low + scaled * (high - low)
    return np.clip(env_action, low, high)


def visualize(agent_name: str, env_name: str, run_id: str, load_option: str, rank=None):
    config = _build_config(agent_name, env_name, run_id, rank=rank)

    base_root = _get_base_root_from_load_option(agent_name, env_name, load_option)
    run_dir = _resolve_run_dir(base_root, agent_name, env_name, run_id, rank=rank)
    checkpoint_path = _resolve_checkpoint_path(run_dir, agent_name, load_option)

    env_kwargs = config["env"].get("kwargs", {}) or {}
    max_episode_steps = int(config["env"].get("max_episode_steps", 0))
    num_episodes = int(config.get("eval", {}).get("eval_episodes", 3))
    seed = int(config["train"].get("seed", 42))
    if rank is not None:
        seed += rank

    policy_action_lim = float(config["env"].get("action_lim", 1.0))

    agent = _build_agent(agent_name, config)
    agent.load_model(str(checkpoint_path))

    env = make_env_demo(
        env_name,
        max_episode_steps=max_episode_steps,
        **env_kwargs,
    )

    print(f"Loaded agent      : {agent_name}")
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
                if agent_name == "SAC":
                    policy_action = agent.act(obs, deterministic=True)
                else:
                    policy_action, _, _ = agent.act(obs, deterministic=True)

                env_action = _policy_to_env_action(
                    env=env,
                    policy_action=policy_action,
                    policy_action_lim=policy_action_lim,
                )

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

    agent = _normalize_agent_name(args.agent)
    env_name = args.env
    run_id = args.runid
    load_option = args.loadOption
    rank = args.rank

    if run_id is None:
        raise ValueError("'--runid' flag must be provided.")

    visualize(
        agent_name=agent,
        env_name=env_name,
        run_id=run_id,
        load_option=load_option,
        rank=rank,
    )

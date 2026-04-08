import argparse
import sys
from pathlib import Path
import numpy as np
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from envs.env import make_env_demo
from agents.SAC import SAC_Agent
from agents.PPO import PPO_Agent
from utils.helper import loadConfig, dir_exist, file_exist

def getParser():
    '''
    # Example Usage
    # Load best agent
    python utils/visualizer.py --agent SAC --env Hopper-v5 --runid 20260408_163845 --loadOption best
    # Load final agent
    python utils/visualizer.py --agent SAC --env Hopper-v5 --runid 20260408_163845 --loadOption final
    # Load agent at checkpoint 100000
    python utils/visualizer.py --agent SAC --env Hopper-v5 --runid 20260408_163845 --loadOption checkpoint_100000
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, default="SAC")
    parser.add_argument("--env", type=str, default="Hopper-v5")
    parser.add_argument("--runid", type=str, default=None)
    parser.add_argument("--loadOption", type=str, default="best")
    return parser.parse_args()

def _normalize_agent_name(agent_name: str) -> str:
    agent_name = agent_name.strip().upper()
    if agent_name not in {"SAC", "PPO"}:
        raise ValueError(f"Unsupported agent '{agent_name}'. Only SAC and PPO are supported.")
    return agent_name

def _resolve_run_dir(agent_name: str, env_name: str, run_id: str) -> Path:
    run_parent = PROJECT_ROOT / "results"
    candidate_roots = [
        run_parent / "best" / agent_name / env_name,
        run_parent / "models" / agent_name / env_name,
        run_parent / "checkpoints" / agent_name / env_name,
    ]
    candidate_dirs = []
    for root in candidate_roots:
        if not dir_exist(str(root)):
            continue
        candidate_dirs.extend([
            root / run_id,
            root / f"{agent_name}_{env_name}_{run_id}",
        ])
        candidate_dirs.extend(sorted(root.glob(f"*{run_id}*")))
    seen = set()
    unique_existing = []
    for path in candidate_dirs:
        resolved = str(path.resolve()) if path.exists() else str(path)
        if resolved in seen:
            continue
        seen.add(resolved)
        if dir_exist(str(path)):
            unique_existing.append(path)
    if not unique_existing:
        searched = "\n".join(str(root) for root in candidate_roots)
        raise FileNotFoundError(
            "Could not find the run directory. Checked under:\n"
            f"{searched}\n\n"
            f"Expected something like: {agent_name}_{env_name}_{run_id}"
        )
    if len(unique_existing) > 1:
        matches = "\n".join(str(path) for path in unique_existing)
        raise FileExistsError(
            "Multiple run directories matched the provided --runid. "
            "Please pass a more specific run id or full folder name:\n"
            f"{matches}"
        )
    return unique_existing[0]

def _resolve_checkpoint_path(agent_name: str, run_dir: Path, load_option: str) -> Path:
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

def _build_config(agent_name: str, env_name: str):
    config_path = PROJECT_ROOT / "configs" / f"{agent_name}.yaml"
    config = loadConfig(str(config_path))
    config["env"]["name"] = env_name
    config["dir"]["root"] = str(PROJECT_ROOT)
    return config

def _build_agent(agent_name: str, config):
    if agent_name == "SAC":
        return SAC_Agent(config)
    if agent_name == "PPO":
        return PPO_Agent(config)
    raise ValueError(f"Unsupported agent '{agent_name}'")

def _policy_to_env_action(env, policy_action):
    policy_action = np.asarray(policy_action, dtype=np.float32)
    low = env.action_space.low
    high = env.action_space.high
    action_scale = 0.5 * (high - low)
    action_bias = 0.5 * (high + low)
    env_action = action_bias + action_scale * policy_action
    return np.clip(env_action, low, high)

def visualize(agent_name: str, env_name: str, run_id: str, load_option: str):
    config = _build_config(agent_name, env_name)
    run_dir = _resolve_run_dir(agent_name, env_name, run_id)
    checkpoint_path = _resolve_checkpoint_path(agent_name, run_dir, load_option)
    env_kwargs = config["env"].get("kwargs", {}) or {}
    max_episode_steps = int(config["env"].get("max_episode_steps", 0))
    num_episodes = int(config.get("eval", {}).get("eval_episodes", 3))
    seed = int(config["train"].get("seed", 42))
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
    print(f"Render mode       : human")
    print(f"Episodes          : {num_episodes}")
    try:
        for ep in range(num_episodes):
            obs, _ = env.reset(seed=seed + 1000 + ep)
            done = False
            ep_return = 0.0
            ep_len = 0
            env.render()
            while not done:
                if agent_name == "SAC":
                    policy_action = agent.act(obs, deterministic=True)
                else:
                    policy_action, _, _ = agent.act(obs, deterministic=True)

                env_action = _policy_to_env_action(env, policy_action)
                obs, reward, terminated, truncated, info = env.step(env_action)
                done = terminated or truncated
                ep_return += reward
                ep_len += 1
                env.render()
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
    if run_id is None:
        raise ValueError("'--runid' flag must be provided.")
    visualize(
        agent_name=agent,
        env_name=env_name,
        run_id=run_id,
        load_option=args.loadOption,
    )

# NOTE : Example run 
# BEST       : python utils/visualizer.py --agent SAC --env Hopper-v5 --runid 20260408_163845 --loadOption best
# FINAL      : python utils/visualizer.py --agent SAC --env Hopper-v5 --runid 20260408_163845 --loadOption final 
# CHECKPOINT : python utils/visualizer.py --agent SAC --env Hopper-v5 --runid 20260408_163845 --loadOption checkpoint_100000 

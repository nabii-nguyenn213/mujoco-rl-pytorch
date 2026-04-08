# Soft Actor Critic and Proximal Policy Optimization Implementation with PyTorch 
This repo implements the **Soft Actor-Critic** (SAC) and **Proximal Policy Optimization** (PPO) algorithms using [PyTorch](https://pytorch.org/) on **Mujoco** environments in [Gymnasium](https://gymnasium.farama.org/environments/mujoco/)

# Environments 

| | [Ant](https://gymnasium.farama.org/environments/mujoco/ant/) | [HalfCheetah](https://gymnasium.farama.org/environments/mujoco/half_cheetah/) | [Hopper](https://gymnasium.farama.org/environments/mujoco/hopper/) | [Humanoid](https://gymnasium.farama.org/environments/mujoco/humanoid/) |
| --- | ---------------- | ------------------ | --------------------------------------- | --------------- |
| Visualization |![](assets/ant.gif) | ![](assets/half_cheetah.gif) | ![](assets/hopper.gif)| ![](assets/humanoid.gif) |
| Action Space | (8,) | (6,) | (3,) | (17,) |
| Observation Space| (105,) | (17,) | (11,) | (348,) |

| | [Humanoid Standup](https://gymnasium.farama.org/environments/mujoco/humanoid_standup/) | [Inverted Double Pendulum](https://gymnasium.farama.org/environments/mujoco/inverted_double_pendulum/#) | [Inverted Pendulum](https://gymnasium.farama.org/environments/mujoco/inverted_pendulum/) | [Pusher](https://gymnasium.farama.org/environments/mujoco/pusher/) |
| --- | ------ | ---------------- | --------------------------------------- | --------------- |
| Visualization | ![](assets/humanoid_standup.gif) | ![](assets/inverted_double_pendulum.gif) | ![](assets/inverted_pendulum.gif)| ![](assets/pusher.gif) |
| Action Space | (17,) | (1,) | (1,) | (7,) |
| Observation Space| (348,) | (9,) | (4,) | (23,) |

| | [Reacher](https://gymnasium.farama.org/environments/mujoco/reacher/) | [Swimmer](https://gymnasium.farama.org/environments/mujoco/swimmer/) | [Walker2D](https://gymnasium.farama.org/environments/mujoco/walker2d/) | 
| --- | ------ | ---------------- | --------------------------------------- | 
| Visualization | ![](assets/reacher.gif) | ![](assets/swimmer.gif) | ![](assets/walker2d.gif)|
| Action Space | (2,) | (2,) | (6,) |
| Observation Space| (10,) | (8,) | (17,) |

# Dependencies & Installation 

- Python version : 3.10.20 
- Libraries : 
    - gymnasium==1.2.3
    - gymnasium[mujoco]==1.2.3
    - numpy==2.4.4
    - omegaconf==2.3.0
    - PyYAML==6.0.3
    - torch==2.11.0
    - tensorboard==2.20.0 (Optional)
    - mpi4py==4.1.1 (Optional)
- Installation: 
```bash
pip install -r requirements.txt 
```
- If you use [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) virtual environment
```bash
conda create -n mujoco_rl python=3.10.20
conda activate mujoco_rl 
pip install -r requirements.txt 
``` 
- If you use [venv](https://docs.python.org/3/library/venv.html) 
    - **Windows** : 
    ```bash 
    py -3.10 -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt
    ```
    - **Linux/MacOS**: 
    ```bash 
    python3.10 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```
- **Verify Setup** (Optional) : After installing the dependencies, run the following command to verify that the environment is configured correctly:
```bash 
python utils/verify_setup.py
```

# Usage 
## Train 

## Description of Configuration Parameters 

## Tensorboard 

# Results 

# Demonstration


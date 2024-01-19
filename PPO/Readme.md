# Proximal Policy Optimization (PPO) Implementations

## Environment
The environment used in these PPO implementations is created by Sven Niederberger and is based on the LunarLander environment by OpenAI. For more details, refer to [gym-rocketlander](https://github.com/EmbersArc/gym-rocketlander).

## PPO Implementations

#### Files:
- `ppo_torch.py`: Neural network architecture for PPO.
- `ppo_main.py`: Main script to execute the PPO algorithm.


## PPO Description

Proximal Policy Optimization (PPO) is an on-policy implementation for the RocketLander environment utilizes an actor-critic architecture, with the actor producing a probability distribution for actions, and the critic estimating state values. Experiences are stored in a memory buffer, and the actor and critic are updated iteratively based on advantages calculated from these experiences. 

## Usage

Ensure you have the necessary dependencies installed before running the scripts. You can install them using:

```bash
pip install -r requirements.txt

# Deep Deterministic Policy Gradient (DDPG) Implementations

## Environment
The environment used in these DDPG implementations is created by Sven Niederberger and is based on the LunarLander environment by OpenAI. For more details, refer to [gym-rocketlander](https://github.com/EmbersArc/gym-rocketlander).

## DDPG Implementations

#### Files:
- `buffer.py`: Implementation of the replay buffer.
- `ddpg_torch.py`: Neural network architecture for DDPG.
- `main_ddpg.py`: Main script to execute the DDPG algorithm.
- 'env.py' : Rendering the environmnet and step method implementation.

## DDPG Description

Deep Deterministic Policy Gradient (DDPG) is an off-policy algorithm specifically designed for handling continuous action spaces. Unlike on-policy algorithms that learn solely from the current policy, DDPG utilizes a replay buffer to store past experiences and learn from them.

## Usage

Ensure you have the necessary dependencies installed before running the scripts. You can install them using:

```bash
pip install -r requirements.txt

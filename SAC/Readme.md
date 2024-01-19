# Soft Actor-Critic (SAC) Implementations

## Environment
The environment used in these SAC implementations is created by Sven Niederberger and is based on the LunarLander environment by OpenAI. For more details, refer to [gym-rocketlander](https://github.com/EmbersArc/gym-rocketlander).

## SAC Implementations

### Without PTAN 
[reference: https://www.youtube.com/watch?v=ioidsRlf79o]

#### Files:
- `buffer.py`: Implementation of the replay buffer.
- `network.py`: Neural network architecture for SAC.
- `sac2.py`: SAC algorithm implementation without PTAN.
- `main.py`: Main script to execute the SAC algorithm without PTAN.

### With PTAN

#### Files:
- `sac3.py`: SAC algorithm implementation with PTAN.
- `main2.py`: Main script to execute the SAC algorithm with PTAN.

### Other Variation (Attempt)

- `sac.py`: A single variation (attempt) of the SAC algorithm.

## SAC Description

Soft Actor-Critic (SAC) is an off-policy actor-critic deep reinforcement learning algorithm designed for continuous action spaces. It aims to maximize the expected cumulative reward while incorporating an entropy term to encourage exploration.

## Usage

Ensure you have the necessary dependencies installed before running the scripts. You can install them using:

```bash
pip install -r requirements.txt

# Falcon 9 Rocket Landing with Reinforcement Learning

## Overview

This repository presents the culmination of our research at Northeastern University, focusing on the development and optimization of a Reinforcement Learning (RL)-based control system for the SpaceX Falcon 9 rocket lander. The primary objective is to achieve autonomous and precise landings on designated target zones, with the ultimate goal of reducing mission costs and enhancing the reusability of space vehicles.

## Contents

- **Code:** The repository houses implementations of three RL algorithms: Deep Deterministic Policy Gradient (DDPG), Proximal Policy Optimization (PPO), and Soft Actor-Critic (SAC). These algorithms are applied to the Rocket Lander environment, a physics-based simulation framework designed for studying the mechanics and dynamics of rocket landings.

- **RocketLander Environment:** Leveraging the Box2D physics engine within the OpenAI Gym interface, this simulation encompasses a comprehensive set of state variables, action spaces, and a reward system. The goal is to provide an accurate representation of the challenges and intricacies involved in landing a rocket.

- **Documentation:** Detailed documentation covers the dynamics and challenges of the environment, state and action spaces, the reward system, and the physics simulation. It also includes related work, showcasing the advancements and current state-of-the-art in RL-based rocket landing.

## Getting Started

To dive into the project:

1. Clone the repository:

   ```bash
   git clone https://github.com/ghime-u/Falcon9_RL.git
   cd Falcon9_RL

## Results

Quantitative results for each RL algorithm are detailed in the documentation. A comparison table summarizes key metrics for DDPG, PPO, and SAC, providing insights into their learning rates, training steps, stability, and success rates.

## Acknowledgments

We extend our gratitude to Phil Zhang for his invaluable RL tutorials and Sven Niederberger for the custom rocket landing environment based on OpenAI's LunarLander. The OpenAI Gym and the wider ML community have been instrumental in shaping our research.

## References

This repository builds upon existing research and implementations. Refer to the provided references for a deeper understanding of the RL algorithms and related work.

## Contact

For inquiries or further information, please contact:

    Ravina Lad
        Email: lad.ra@northeastern.edu

    Utkarsh Ghime
        Email: ghime.u@northeastern.edu

## License

This project is licensed under the MIT License. Feel free to use, modify, and distribute the code as per the terms of the license. We welcome collaboration and contributions from the community to advance the field of RL in rocket landing applications.

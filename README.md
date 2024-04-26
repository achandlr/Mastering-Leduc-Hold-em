# Mastering Leduc Hold'em: A Comparative Study of CFR and Reinforcement Learning

This project is an extension of the RLCard library, focusing on a comparative study of Counterfactual Regret Minimization (CFR) and Reinforcement Learning (RL) algorithms for the game of Leduc Hold'em. We have implemented 18 different agents, including various CFR and RL agents, and compared their performance against each other.

## Our Contributions
### We have added the following agents to the RLCard library:

#### CFR Agents:
- CFR
- Monte Carlo CFR (Variants: Outcome Sampling, External Sampling, Average Sampling)
- CFR+
- Deep CFR

#### Naive Agents:
- Check Fold (Ultra-Passive)
- Raise Call (Ultra-Aggresive)

#### SARSA Agents:
- SARSA
- Expected Sarsa
- N-Step SARSA
- SARSA(λ)
- True Online SARSA(λ)
- Deep SARSA(λ)

#### Other Agents:
- Q-Learning
- REINFORCE

All of our agent implementations can be found in the rlcard/agents/our_agents directory.

## Installation
Our installation follows the basic process outlined in the RLCard readme. Please follow the following process.

- git clone our repository
- Ensure that you have **Python 3.8+** and **pip** installed.

The following steps are
```
pip3 install rlcard
cd rlcard
pip3 install -e .
```

### Training Agents
To train our agents on Leduc Hold'em, run the following command:
```
python examples/train_agents.py
```

### Evaluating Agents
To evaluate our agents and visualizae the comparative agent performance on the game of Leduc Hold'em, run the following command:

```
python examples/evaluate_agents.py
```

This will create a round-robin tournament where each agent plays against every other agent. The evaluation metrics include:

- Average Big Blind per Hand (BBPH)
- Elo rating of each agent 

### Acknowledgements

This project is built upon the RLCard library. We would like to express our gratitude to the RLCard team for their excellent work and for providing a platform for researching reinforcement learning in card games.
Please refer to the original RLCard README below for more information on the library, including installation instructions, available environments, and supported algorithms.
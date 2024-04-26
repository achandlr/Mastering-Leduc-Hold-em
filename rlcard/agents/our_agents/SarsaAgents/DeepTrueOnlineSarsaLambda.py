import numpy as np
import os
import pickle
import torch
import torch.nn as nn
from rlcard.agents.our_agents.SarsaAgents.SarsaBaseAgent import SarsaBaseAgent


class QNetwork(nn.Module):
    """
    QNetwork class represents a deep Q-network for reinforcement learning.

    Args:
        state_size (int): The size of the input state.
        num_actions (int): The number of possible actions.

    Attributes:
        layers (nn.Sequential): The layers of the neural network.

    Methods:
        forward(state): Performs a forward pass through the network.

    """

    def __init__(self, state_size, num_actions):
        super(QNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )

    def forward(self, state):
        """
        Performs a forward pass through the network.

        Args:
            state (torch.Tensor): The input state.

        Returns:
            torch.Tensor: The output of the network.

        """
        return self.layers(state)

class DeepTrueOnlineSarsaLambdaAgent(SarsaBaseAgent):
    """
    DeepTrueOnlineSarsaLambdaAgent is an implementation of the True Online Sarsa(λ) algorithm with a deep Q-network.

    Args:
        config (dict): Configuration dictionary containing hyperparameters.

    Attributes:
        lam (float): Lambda value for eligibility traces.
        device (torch.device): Device to run the agent on (CPU or GPU).
        q_network (QNetwork): Deep Q-network used for value estimation.
        optimizer (torch.optim.Optimizer): Optimizer used for updating the Q-network.

    """

    def __init__(self, config):
        super().__init__(config)
        self.lam = config.get('lam', 0.9)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = QNetwork(self.num_state_features, self.num_actions).to(self.device)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=config.get('alpha', 0.001))

    def save(self):
        """
        Save the Q-network and optimizer parameters to a file.
        """
        items_to_save = {"q_network": self.q_network, "optimizer": self.optimizer}
        super().save(items_to_save)

    def save_only_best(self, reward):
        """
        Save the Q-network and optimizer parameters to a file if the reward is the best so far.

        Args:
            reward (float): The reward obtained by the agent.

        """
        items_to_save = {"q_network": self.q_network, "optimizer": self.optimizer}
        super().save_only_best(reward, items_to_save)

    def load(self):
        """
        Load the Q-network and optimizer parameters from a file.
        """
        files_to_load = ["q_network", "optimizer"]
        super().load(files_to_load)

    def select_action(self, state, use_epsilon_greedy=False):
        """
        Select an action to take based on the current state.

        Args:
            state (dict): The current state of the environment.
            use_epsilon_greedy (bool): Whether to use epsilon-greedy exploration.

        Returns:
            chosen_action: The action chosen by the agent.

        """
        legal_actions = list(state['legal_actions'].keys())
        state_tensor = torch.tensor(state['obs'], dtype=torch.float32).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor).squeeze().detach().cpu().numpy()
        if use_epsilon_greedy:
            val_from_zero_to_one = np.random.rand()
            curr_epsilon = self.get_epsilon()
            if val_from_zero_to_one < curr_epsilon:
                chosen_action = np.random.choice(legal_actions)
            else:
                q_values_legal = q_values[legal_actions]
                chosen_action = legal_actions[np.argmax(q_values_legal)]
        else:
            q_values_legal = q_values[legal_actions]
            chosen_action = legal_actions[np.argmax(q_values_legal)]
        
        return chosen_action

    def train(self):
        """
        Train the agent using the True Online Sarsa(λ) algorithm.
        """
        self.curr_episode_index += 1
        for player_id in range(self.env.num_players):
            self.env.reset()
            next_state = None
            done = False
            z = np.zeros(self.num_state_features)
            next_action_selected = None
            while not done:
                if next_state is None:
                    state = self.env.get_state(player_id)
                else:
                    state = next_state

                if next_action_selected is None:
                    action_selected = self.select_action(state, use_epsilon_greedy=True)
                else:
                    action_selected = next_action_selected

                self.env.step(action_selected)
                next_state = self.env.get_state(player_id)
                done = self.env.is_over()

                if done:
                    reward = self.env.get_payoffs()
                    R = reward[player_id]
                else:
                    R = 0

                if not done:
                    # Choose A' ∼ π(·|S') or near greedily from S' using w (pseudocode line)
                    next_action_selected = self.select_action(next_state, use_epsilon_greedy=True)
                else:
                    next_action_selected = None
                    pass # TODO: is this correct? is next_action_selected still used pointing to an invalid variable


                state_tensor = torch.tensor(state['obs'], dtype=torch.float32).unsqueeze(0).to(self.device)
                next_state_tensor = torch.tensor(next_state['obs'], dtype=torch.float32).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor).squeeze()
                next_q_values = self.q_network(next_state_tensor).squeeze()
                Q = q_values[action_selected] # BUG Q is not used. this is likely a bug
                # Q_next = next_q_values[next_action_selected] if not done else torch.tensor(0.0)
                Q_next = torch.tensor(0.0) if done else next_q_values[next_action_selected]  # Set Q_next to zero for terminal states

                # TODO: likley bugs here. need check pseudocode and wee if we should use q only or q. I took out q_old but maybe i should reintroduce it. look at the pseudocode to decide
                # δ ← R + γQ' - Q_old (pseudocode line)
                delta = R + self.gamma * Q_next - Q
                # q_old = Q.item()

                # Update eligibility traces
                z = z * self.gamma * self.lam
                z[action_selected] += 1

                # Compute the loss using the TD error and eligibility traces
                loss = delta * z[action_selected]

                # Scale and normalize the eligibility traces if needed
                # For example: self.eligibility_traces /= self.eligibility_traces.norm()
                self.optimizer.zero_grad()
                loss.backward(retain_graph=True) # TODO: verify this is correct solution
                self.optimizer.step()
                # q_old = Q_next.item()
                

                if done:
                    break

        return

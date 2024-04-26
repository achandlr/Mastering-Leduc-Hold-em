import numpy as np
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from rlcard.agents.our_agents.BaseAgent import BaseAgent

'''
Note: This code is a modified copy of my Project Four Submission Code as part of CS394
'''

class PiApproximationWithNN(nn.Module):
    def __init__(self, state_dims, num_actions, alpha):
        """
        PiApproximationWithNN is a class that represents a policy approximation model using a neural network.

        Args:
            state_dims (int): The number of dimensions of the state space.
            num_actions (int): The number of possible actions.
            alpha (float): The learning rate.

        Attributes:
            state_dims (int): The number of dimensions of the state space.
            num_actions (int): The number of possible actions.
            alpha (float): The learning rate.
            network (nn.Sequential): The neural network model.
            optim (torch.optim.Adam): The optimizer for the neural network model.
        """
        super(PiApproximationWithNN, self).__init__()
        self.state_dims = state_dims
        self.num_actions = num_actions
        self.alpha = alpha
        self.network = nn.Sequential(
            nn.Linear(state_dims, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )
        self.optim = optim.Adam(self.network.parameters(), lr=alpha, betas=(0.9, 0.999))

    def forward(self, states, return_prob=True):
        """
        Forward pass of the neural network model.

        Args:
            states (torch.Tensor or list): The input states.
            return_prob (bool, optional): Whether to return the action probabilities or not. Defaults to True.

        Returns:
            torch.Tensor: The action probabilities if return_prob is True, otherwise the action index.
        """
        if not isinstance(states, torch.Tensor):
            states = torch.tensor(states, dtype=torch.float32)
        output = self.network(states)
        if return_prob:
            return torch.softmax(output, dim=-1)
        else:
            return torch.softmax(output, dim=-1).argmax().item()

    def __call__(self, s):
        """
        Callable method to select an action based on the current state.

        Args:
            s (torch.Tensor or list): The current state.

        Returns:
            int: The selected action.
        """
        probs = self.forward(s)
        action = np.random.choice(range(self.num_actions), p=probs.detach().numpy())
        return action

    def update(self, states, actions_taken, gamma_t, delta):
        """
        Update the neural network model based on the given states, actions, and rewards.

        Args:
            states (torch.Tensor or list): The states.
            actions_taken (list): The actions taken.
            gamma_t (float): The discount factor.
            delta (float): The TD error.

        Returns:
            None
        """
        self.optim.zero_grad()
        if not isinstance(states, torch.Tensor):
            states_tensor = torch.tensor(states, dtype=torch.float32)
        else:
            states_tensor = states
        actions_taken_tensor = torch.tensor(actions_taken, dtype=torch.long)
        gamma_t_tensor = torch.tensor(gamma_t, dtype=torch.float32)
        delta_tensor = torch.tensor(delta, dtype=torch.float32)

        action_probs = self.forward(states_tensor, return_prob=True)
        action_prob_taken = action_probs[actions_taken_tensor.item()]
        log_prob_action_taken = torch.log(action_prob_taken)
        policy_gradient = log_prob_action_taken * delta_tensor * gamma_t_tensor
        loss = -policy_gradient
        
        loss.backward()
        self.optim.step()
        return

class VApproximationWithNN(nn.Module):
    def __init__(self, state_dims, alpha):
        """
        state_dims: the number of dimensions of state space
        alpha: learning rate
        """
        super(VApproximationWithNN, self).__init__()
        self.state_dims = state_dims
        self.alpha = alpha
        # assert alpha <= 3 * (10**-4)
        self.network = nn.Sequential(
            nn.Linear(state_dims, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1) 
        )
        self.optim = optim.Adam(self.network.parameters(), lr=alpha, betas=(0.9, 0.999))

        return 

    def forward(self, states) -> float:
        if not isinstance(states, torch.Tensor):
            states = torch.tensor(states, dtype=torch.float32)
        state_val = self.network(states)
        return state_val

    def update(self, states, delta):
        self.optim.zero_grad()
        # check if states is a tensor
        if not isinstance(states, torch.Tensor):
            states_tensor = torch.tensor(states, dtype=torch.float32)
        else:
            states_tensor = states
        delta_tensor = torch.tensor(delta, dtype=torch.float32)
        output_val = self.forward(states_tensor).squeeze()
        loss = F.mse_loss(delta_tensor, output_val)     
        loss.backward()
        self.optim.step()
        return
    
class ReinforceAgent(BaseAgent):

    def __init__(self, env, model_path, perform_logging=False, load_saved_model=True, save_model=True):
        """
        Initializes the Reinforce agent.

        Args:
            env (object): The environment object.
            model_path (str): The path to save the model.
            perform_logging (bool, optional): Whether to perform logging. Defaults to False.
            load_saved_model (bool, optional): Whether to load a saved model. Defaults to True.
            save_model (bool, optional): Whether to save the model. Defaults to True.
        """
        super().__init__(env, model_path, perform_logging, load_saved_model, save_model)
        self.gamma = 1 # 0.8 # TODO: experiment with other gammas
        self.curr_episode_index = 0
        self.num_state_features = 36
        self.num_action_features = 4
        self.curr_episode_index = 0
        self.pi_alpha = 3e-3 # TODO: Increase if needed
        self.v_alpha = 3e-3 # TODO: Increase if needed
        self.best_reward = -np.inf
        self.policy = PiApproximationWithNN(self.num_state_features, self.num_action_features, self.pi_alpha)
        self.value_approximator = VApproximationWithNN(self.num_state_features, self.v_alpha)

    @staticmethod
    def calc_monte_carlo_returns(rewards, gamma):
        T = len(rewards)
        returns = [0 for _ in range(T)] 
        G = 0  
        for t in reversed(range(T)):
            G = gamma * G + rewards[t]
            returns[t] = G
        return returns
    
    def select_action(self, state):
        state_as_tensor = torch.tensor(state['obs'], dtype=torch.float32) 
        action = self.policy(state_as_tensor)
        return action
    
    def eval_step(self, state):
        chosen_action = self.select_action(state)
        info = {} # Note: not storing any info for now unlike other classes but I don't think this is a problem because info should not be used in this code
        return chosen_action, info
    
    def save(self):
        items_to_save = {"policy": self.policy, "value_approximator": self.value_approximator}
        super().save(items_to_save)

    def save_only_best(self, reward):
        items_to_save = {"policy": self.policy, "value_approximator": self.value_approximator}
        super().save_only_best(reward, items_to_save)

    def load(self):
        files_to_load = ["policy", "value_approximator"]
        super().load(files_to_load)
        
    def train(self):
        self.curr_episode_index += 1
        list_of_g_0_each_episode = []
        for player_id in range(self.env.num_players):
            self.env.reset()
            curr_state = torch.tensor(self.env.get_state(player_id)['obs'], dtype=torch.float32) 
            done = False

            states = []
            actions = []
            rewards = []

            while not done:
                action = self.policy(curr_state)
                self.env.step(action)
                next_state = torch.tensor(self.env.get_state(player_id)['obs'], dtype=torch.float32) 
                done = self.env.is_over()
                if done:
                    reward_both_players = self.env.get_payoffs()
                    reward = reward_both_players[player_id]
                else:
                    reward = 0
                states.append(curr_state)
                actions.append(action)
                rewards.append(reward)
                curr_state = next_state

            monte_carlo_returns = ReinforceAgent.calc_monte_carlo_returns(rewards, self.gamma)
            list_of_g_0_each_episode.append(monte_carlo_returns[0])

            for t in range(len(states)):                
                state_tensor_value_estimation = states[t] # torch.tensor(states[t], dtype=torch.float32) 
                V_est = self.value_approximator.forward(state_tensor_value_estimation).item() 
                TD_error = monte_carlo_returns[t] - V_est
                self.value_approximator.update(states[t], TD_error)
                self.policy.update(states[t], actions[t], self.gamma ** t, TD_error)
        return
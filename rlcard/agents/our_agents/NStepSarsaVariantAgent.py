import numpy as np
import os
from collections import defaultdict
import pickle
from rlcard.utils import remove_illegal
from rlcard.agents.our_agents.BaseAgent import BaseAgent

'''
Updates are calculated when the length of the states list equals n or when the episode ends.
The target (G) is calculated as the sum of discounted rewards from the current state to the end of the episode.
The Q-value of the first state-action pair in the current batch is updated using the calculated target and the learning rate α.
The update rule is: Q(S_0, A_0) += α * (target - Q(S_0, A_0)), where S_0 and A_0 are the first state and action in the current batch.
The agent maintains lists of states, actions, and rewards for each batch of size n or until the end of the episode.
At each time step, the agent appends the current state, action, and reward to the respective lists.
When the batch size reaches n or the episode ends, the agent updates the Q-value of the first state-action pair in the batch using the calculated target.
After updating, the agent clears the lists of states, actions, and rewards to start a new batch.
Action selection is performed using an ε-greedy policy based on the current Q-values.

It is an MC method because it updates the Q-values based on the complete returns (sum of discounted rewards) from each state until the end of the episode.
The updates are performed in batches of size n or until the end of the episode, making it a batch MC method.
'''

class NStepSarsaVariantAgent(BaseAgent):
    def __init__(self, env, model_path, perform_logging=False, load_saved_model=True, save_model=True, n=2):
        """
        Initializes an instance of the NStepSarsaVariantAgent class.

        Parameters:
        - env (object): The environment object.
        - model_path (str): The path to the model.
        - perform_logging (bool): Flag indicating whether to perform logging. Default is False.
        - load_saved_model (bool): Flag indicating whether to load a saved model. Default is True.
        - save_model (bool): Flag indicating whether to save the model. Default is True.
        - n (int): The number of steps for n-step SARSA. Default is 2.
        """
        super().__init__(env, model_path, perform_logging, load_saved_model, save_model)
        self.q_values = defaultdict(self.default_q_value)
        self.gamma = 0.8
        self.alpha = 1e-1
        self.eps_start = 1
        self.lowest_epsilon_value = 1e-2
        self.eps_decay_factor = 0.999
        self.curr_episode_index = 0
        self.n = n

    def default_q_value(self):
        return np.zeros(self.env.num_actions)

    def eval_step(self, state):
        chosen_action, _ = self.select_action(state, use_epsilon_greedy=False)
        info = {}  # Note: not storing any info for now unlike other classes but I don't think this is a problem because info should not be used in this code
        return chosen_action, info

    def save(self):
        items_to_save = {"q_values": self.q_values}
        super().save(items_to_save)

    def save_only_best(self, reward):
        items_to_save = {"q_values": self.q_values}
        super().save_only_best(reward, items_to_save)
        
    def load(self):
        # file_path = os.path.join(self.model_path, "q_values.pkl")
        # self.q_values = pickle.load(open(file_path, "rb"))
        files_to_load = ["q_values"]
        super().load(files_to_load)

    def get_epsilon(self):
        epsilon = max(self.eps_start * (self.eps_decay_factor ** self.curr_episode_index), self.lowest_epsilon_value)
        return epsilon
    
    def select_action(self, state, use_epsilon_greedy=False):
        state_obs = tuple(state['obs'])
        legal_actions = list(state['legal_actions'].keys())
        val_from_zero_to_one = np.random.rand()
        if use_epsilon_greedy:
            curr_epsilon = self.get_epsilon()
        else:
            curr_epsilon = 0  # This value indicates that we are not using epsilon greedy. Will only pick random action if the state is not in q_values
        if val_from_zero_to_one < curr_epsilon or state_obs not in self.q_values or len(self.q_values[state_obs]) == 0:
            chosen_action = np.random.choice(legal_actions)
        else:
            # Make all illegal actions have -inf so that they are not selected
            masked_arr = np.full(self.q_values[state_obs].shape, -np.inf)
            masked_arr[legal_actions] = self.q_values[state_obs][legal_actions]
            chosen_action = np.argmax(masked_arr)
        return chosen_action, state_obs

    def observe_state_and_actions(self, curr_player_id):
        """
        Observe the current state and legal actions for a player.

        Args:
            curr_player_id (int): The current player's ID.

        Returns:
            tuple: Tuple containing the observed state and list of legal actions.
        """
        state = self.env.get_state(curr_player_id)
        state_matrix = state['obs']
        legal_actions = list(state['legal_actions'].keys())
        observed_state_tuple = tuple(state_matrix)
        return observed_state_tuple, legal_actions

    # # This works empirically, but not directly following pseudocode so am commenting out
    def update_q_values(self, states, actions, rewards):
        states_tuples = [tuple(state['obs']) for state in states]
        next_state_tuple = tuple(states[-1]['obs'])

        for state_tuple in states_tuples:
            if state_tuple not in self.q_values:
                self.q_values[state_tuple] = np.zeros(self.env.num_actions)
        if next_state_tuple not in self.q_values:
            self.q_values[next_state_tuple] = np.zeros(self.env.num_actions)

        target = 0.0
        for i in range(len(rewards) - 1, -1, -1):
            target = rewards[i] + self.gamma * target

        self.q_values[states_tuples[0]][actions[0]] += self.alpha * (target - self.q_values[states_tuples[0]][actions[0]])

    def train(self):
        # Loop for each episode (2 episdoed per training step, one for each player)
        for player_id in range(self.env.num_players):
            self.env.reset()
            self.curr_episode_index += 1
            states = []
            actions = []
            rewards = []
            next_state = None
            while not self.env.is_over():
                if next_state is None:
                    state = self.env.get_state(player_id)
                else:
                    state = next_state
                states.append(state)

                action, _ = self.select_action(state, use_epsilon_greedy=True)
                actions.append(action)
                self.env.step(action)

                if self.env.is_over():
                    reward = self.env.get_payoffs()[player_id]
                else:
                    reward = 0
                rewards.append(reward)

                if len(states) == self.n or self.env.is_over():
                    self.update_q_values(states, actions, rewards)
                    states = []
                    actions = []
                    rewards = []

                next_state = self.env.get_state(player_id)
        return

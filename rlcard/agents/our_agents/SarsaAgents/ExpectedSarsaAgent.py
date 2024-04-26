import numpy as np
import os
from collections import defaultdict
import pickle
from rlcard.utils import remove_illegal
from rlcard.agents.our_agents.SarsaAgents.SarsaBaseAgent import SarsaBaseAgent

# TODO: identify learning instability. probably need to decrease decays more quickly
# TODO: experiment lowering alpha


class ExpectedSarsaAgent(SarsaBaseAgent):
    """
    An agent that implements the Expected SARSA algorithm for reinforcement learning.

    Parameters:
    - config (dict): A dictionary containing the configuration parameters for the agent.

    Attributes:
    - policy (defaultdict): A dictionary that stores the Q-values for each state-action pair.
    - alpha (float): The learning rate for updating the Q-values.
    - gamma (float): The discount factor for future rewards.

    Methods:
    - update_policy(state, action, reward, next_state): Updates the Q-value for a state-action pair using the Expected SARSA update rule.
    - get_action_probs(state): Calculates the action probabilities for a given state.
    - save(): Saves the agent's policy to a file.
    - save_only_best(reward): Saves the agent's policy to a file only if the given reward is better than the previous best reward.
    - load(): Loads the agent's policy from a file.
    - get_epsilon(): Calculates the current exploration rate (epsilon) based on the episode index.
    - select_action(state, use_epsilon_greedy): Selects an action based on the epsilon-greedy policy.
    - train(): Trains the agent by updating the policy based on the observed rewards and states.
    """
    def __init__(self, config):
        super().__init__(config)
        self.policy = defaultdict(list)
        self.alpha = config.get('alpha', 0.1)

    def update_policy(self, state, action, reward, next_state):
        # Convert the current state and next state to tuples for dictionary key
        state_tuple = tuple(state['obs'])
        next_state_tuple = tuple(next_state['obs'])
        
        # Initialize Q-values for the current state and next state if not present in the policy
        if state_tuple not in self.policy:
            self.policy[state_tuple] = np.zeros(self.env.num_actions)
        if next_state_tuple not in self.policy:
            self.policy[next_state_tuple] = np.zeros(self.env.num_actions)
        
        # Calculate the expected Q-value for the next state using the action probabilities
        next_action_probs = self.get_action_probs(next_state)
        expected_q_next_state = np.sum(next_action_probs * self.policy[next_state_tuple])
        
        # Update the Q-value for the current state-action pair using the Expected SARSA update rule
        # Q(s,a) = Q(s,a) + α * (r + γ * Σ_a' π(a'|s') * Q(s',a') - Q(s,a))
        new_q = reward + (self.gamma * expected_q_next_state)
        self.policy[state_tuple][action] += self.alpha * (new_q - self.policy[state_tuple][action])
        return
            
    def get_action_probs(self, state):
        # Convert the state observation to a tuple for dictionary key
        state_obs = tuple(state['obs'])
        legal_actions = list(state['legal_actions'].keys())
        
        # Get the current exploration rate (epsilon)
        epsilon = self.get_epsilon()
        num_actions = self.env.num_actions
        num_legal_actions = len(legal_actions)
        
        # Initialize the action probabilities array
        action_probs = np.zeros(num_actions)
        
        if state_obs not in self.policy or np.all(self.policy[state_obs] == 0):
            # If the state is not in the policy or all Q-values are zero,
            # assign equal probabilities to legal actions
            action_probs[legal_actions] = 1 / num_legal_actions
        else:
            # Get the Q-values for the state from the policy
            q_values = self.policy[state_obs]
            masked_arr = np.full(q_values.shape, -np.inf)
            masked_arr[legal_actions] = q_values[legal_actions]
            # Find the best action among the legal actions
            chosen_action = np.argmax(masked_arr)
            best_action = chosen_action
            
            # Assign probabilities based on the epsilon-greedy policy
            for action in range(num_actions):
                if action in legal_actions:
                    if action == best_action:
                        # Assign higher probability to the best action
                        action_probs[action] = 1 - epsilon + epsilon / num_legal_actions
                    else:
                        # Assign equal probabilities to other legal actions
                        action_probs[action] = epsilon / num_legal_actions
                else:
                    # Assign zero probability to illegal actions
                    action_probs[action] = 0
        
        # Ensure that the probabilities of illegal actions are zero
        action_probs = remove_illegal(action_probs, legal_actions)
        
        return action_probs

    def save(self):
        items_to_save = {"policy": self.policy}
        super().save(items_to_save)
        # now save cumulative_policy_estimate as pickle file
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        with open(os.path.join(self.model_path, 'policy.pkl'), 'wb') as f:
            pickle.dump(self.policy, f)

    def save_only_best(self, reward):
        items_to_save = {"policy": self.policy}
        super().save_only_best(reward, items_to_save)

    def load(self):
        files_to_load = ["policy"]
        super().load(files_to_load)

    def get_epsilon(self):
        epsilon = max(self.eps_start * (self.eps_decay_factor ** self.curr_episode_index), self.lowest_epsilon_value)
        return (epsilon)

    # select action based on epsilon greedy
    def select_action(self, state, use_epsilon_greedy = False):
        state_obs =  tuple(state['obs'])
        legal_actions = list(state['legal_actions'].keys())
        val_from_zero_to_one = np.random.rand()
        if use_epsilon_greedy:
            curr_epsilon = self.get_epsilon()
        else:
            curr_epsilon = 0 # This value indicates that we are not using epsilon greedy. Will only pick random action if the state is not in policy
        if val_from_zero_to_one < curr_epsilon or state_obs not in self.policy or len(self.policy[state_obs]) == 0:
            chosen_action = np.random.choice(legal_actions)
        else:
            # Make all illegal actions have -inf so that they are not selected
            masked_arr = np.full(self.policy[state_obs].shape, -np.inf)
            masked_arr[legal_actions] = self.policy[state_obs][legal_actions]
            chosen_action = np.argmax(masked_arr)
        return chosen_action

    def train(self):
        self.curr_episode_index += 1
        for player_id in range(self.env.num_players):
            self.env.reset()
            next_state = None
            while not self.env.is_over():
                if next_state is None:
                    state = self.env.get_state(player_id)
                else:
                    state = next_state
                # select action with epsilon greedy while training
                action = self.select_action(state, use_epsilon_greedy=True)
                self.env.step(action)
                if self.env.is_over():
                    reward = self.env.get_payoffs()
                    # Note: This could be catestrophic mistake and lead to q values not being updated correctly. we need to go back here and check if this is a fatal mistake. Is state player dependent or not?
                    reward_curr_player = reward[player_id]
                else:
                    reward_curr_player =  0
                next_state = self.env.get_state(player_id)
                self.update_policy(state, action, reward_curr_player, next_state)
        return

import numpy as np
from collections import defaultdict
from rlcard.agents.our_agents.BaseAgent import BaseAgent

class QLearningAgent(BaseAgent):

    def __init__(self, env, model_path, perform_logging=False, load_saved_model=True, save_model=True):
        """
        Initialize a QLearningAgent object.

        Args:
            env (object): The RL environment.
            model_path (str): The path to save/load the Q-learning model.
            perform_logging (bool, optional): Whether to perform logging. Defaults to False.
            load_saved_model (bool, optional): Whether to load a saved model. Defaults to True.
            save_model (bool, optional): Whether to save the model. Defaults to True.
        """
        super().__init__(env, model_path, perform_logging, load_saved_model, save_model)
        self.q_values = defaultdict(self.default_q_value)
        self.gamma = 0.8
        self.alpha = 1e-1
        self.eps_start = 1
        self.lowest_epsilon_value = 1e-2 
        self.eps_decay_factor = 0.999
        self.curr_episode_index = 0

    def default_q_value(self):
        return np.zeros(self.env.num_actions)

    def update_policy(self, state, action, reward, next_state):
        # Convert the current state and next state to tuples for dictionary key
        state_tuple = tuple(state['obs'])
        next_state_tuple = tuple(next_state['obs'])
        
        # Update the Q-value for the current state-action pair using the Q-learning update rule
        # TODO: verify reward is reward R_t+1
        # Q(s,a) = Q(s,a) + α * (r + γ * max_a' Q(s',a') - Q(s,a))
        max_q_next_state = np.max(self.q_values[next_state_tuple])
        new_q = reward + (self.gamma * max_q_next_state)
        self.q_values[state_tuple][action] += self.alpha * (new_q - self.q_values[state_tuple][action])
        
        return

    def eval_step(self, state):
        chosen_action = self.select_action(state, use_epsilon_greedy=False)
        info = {} # Note: not storing any info for now unlike other classes but I don't think this is a problem because info should not be used in this code
        return chosen_action, info
    
    def save(self):
        items_to_save = {"q_values": self.q_values}
        super().save(items_to_save)

    def save_only_best(self, reward):
        items_to_save = {"q_values": self.q_values}
        super().save_only_best(reward, items_to_save)

    def load(self):
        items_to_load = ["q_values"]
        super().load(items_to_load)

    def get_epsilon(self):
        epsilon = max(self.eps_start * (self.eps_decay_factor ** self.curr_episode_index), self.lowest_epsilon_value)
        return epsilon
    
    # select action based on epsilon greedy
    def select_action(self, state, use_epsilon_greedy = False):
        state_obs =  tuple(state['obs'])
        legal_actions = list(state['legal_actions'].keys())
        val_from_zero_to_one = np.random.rand()
        if use_epsilon_greedy:
            curr_epsilon = self.get_epsilon()
        else:
            curr_epsilon = 0 # This value indicates that we are not using epsilon greedy. Will only pick random action if the state is not in q_values
        if val_from_zero_to_one < curr_epsilon or state_obs not in self.q_values or len(self.q_values[state_obs]) == 0:
            chosen_action = np.random.choice(legal_actions)
        else:
            # Make all illegal actions have -inf so that they are not selected
            masked_arr = np.full(self.q_values[state_obs].shape, -np.inf)
            masked_arr[legal_actions] = self.q_values[state_obs][legal_actions]
            chosen_action = np.argmax(masked_arr)
        return chosen_action

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
                    reward_curr_player = reward[player_id]
                else:
                    reward_curr_player =  0
                next_state = self.env.get_state(player_id)
                self.update_policy(state, action, reward_curr_player, next_state)
        return
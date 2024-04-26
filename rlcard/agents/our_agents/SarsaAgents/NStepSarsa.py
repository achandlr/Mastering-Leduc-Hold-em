import numpy as np
from collections import defaultdict
from rlcard.agents.our_agents.SarsaAgents.SarsaBaseAgent import SarsaBaseAgent

class NStepSarsaAgent(SarsaBaseAgent):
    def __init__(self, config):
        """
        Initializes a new instance of the NStepSarsa class.

        Args:
            config (dict): A dictionary containing the configuration parameters.

        Attributes:
            q_values (defaultdict): A dictionary that stores the Q-values for each state-action pair.
            n (int): The number of steps to look ahead in the N-step SARSA algorithm.
        """
        super().__init__(config)
        self.q_values = defaultdict(self.default_q_value)
        self.n = config.get('n', 2)

    def default_q_value(self):
        return np.zeros(self.env.num_actions)

    def save(self):
        items_to_save = {"q_values": self.q_values}
        super().save(items_to_save)

    def save_only_best(self, reward):
        items_to_save = {"q_values": self.q_values}
        super().save_only_best(reward, items_to_save)


    def load(self):
        files_to_load = ["q_values"]
        super().load(files_to_load)

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

    def update_q_values(self, states, actions, rewards, tau, T):

        states_tuples = [tuple(state['obs']) for state in states]

        for state_tuple in states_tuples:
            if state_tuple not in self.q_values:
                self.q_values[state_tuple] = np.zeros(self.env.num_actions)


        assert tau >= 0 and tau < len(states), "Invalid tau value"
        assert len(states) == len(actions) == len(rewards), "Length of states, actions, and rewards should be equal"
        if tau + self.n < T:
            G = 0
            for i in range(tau + 1, min(tau + self.n + 1, len(rewards))):
                G += (self.gamma ** (i - tau - 1)) * rewards[i]
            if tau + self.n < len(states):
                next_state = states[tau + self.n]
                next_action = actions[tau + self.n]
                G += (self.gamma ** self.n) * self.q_values[tuple(next_state['obs'])][next_action]
        else:
            G = 0
            for i in range(tau + 1, len(rewards)):
                G += (self.gamma ** (i - tau - 1)) * rewards[i]
        state_tuple = tuple(states[tau]['obs'])
        action = actions[tau]
        self.q_values[state_tuple][action] += self.alpha * (G - self.q_values[state_tuple][action])

    def train(self):
        # Note: This is a modified version of my own coding implementations from CS394 (this class).
        self.curr_episode_index += 1
        for player_id in range(self.env.num_players):
            T = float('inf')  
            self.env.reset()
            
            states = []
            actions = []
            rewards = []
            next_state = None       
            t = 0    
            while t < T:
                tau = t - self.n + 1  # (tau is the time whose estimate is being updated)
                if next_state is None:
                    state = self.env.get_state(player_id)
                else:
                    state = next_state
                states.append(state)
                action = self.select_action(state, use_epsilon_greedy=True)
                actions.append(action)
                self.env.step(action)

                done_with_episode = self.env.is_over()
                if done_with_episode:
                    T = t+1
                    reward = self.env.get_payoffs()[player_id]
                else:
                    reward = 0
                rewards.append(reward)
                next_state = self.env.get_state(player_id)
                if tau >= 0:
                    self.update_q_values(states, actions, rewards, tau, T)
                t += 1  
        while tau < T - 1:
            tau += 1
            self.update_q_values(states, actions, rewards, tau, T)
        return
    

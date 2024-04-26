import numpy as np
from collections import defaultdict
from rlcard.agents.our_agents.SarsaAgents.SarsaBaseAgent import SarsaBaseAgent

class SarsaAgent(SarsaBaseAgent):
    def __init__(self, config):
            """
            Initialize the SarsaAgent.

            Args:
                config (dict): A dictionary containing the configuration parameters for the agent.
            """
            super().__init__(config)
            self.policy = defaultdict(list)

    def update_policy(self, state, action, reward, next_state):
        next_action = self.select_action(next_state, use_epsilon_greedy=True)
        state_tuple = tuple(state['obs'])
        next_state_tuple = tuple(next_state['obs'])

        if state_tuple not in self.policy:
            self.policy[state_tuple] = np.zeros(self.env.num_actions)
        if next_state_tuple not in self.policy:
            self.policy[next_state_tuple] = np.zeros(self.env.num_actions)

        new_q = reward + (self.gamma * self.policy[next_state_tuple][next_action])
        self.policy[state_tuple][action] = self.policy[state_tuple][action] + (self.alpha * (new_q - self.policy[state_tuple][action]))
        return
    
    def save(self):
        items_to_save = {"policy": self.policy}
        super().save(items_to_save)

    def save_only_best(self, reward):
        items_to_save = {"policy": self.policy}
        super().save_only_best(reward, items_to_save)

    def load(self):
        files_to_load = ["policy"]
        super().load(files_to_load)

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
                    # TODO: This could be catestrophic mistake and lead to q values not being updated correctly. we need to go back here and check if this is a fatal mistake. Is state player dependent or not?
                    reward_curr_player = reward[player_id]
                else:
                    reward_curr_player =  0
                next_state = self.env.get_state(player_id)
                self.update_policy(state, action, reward_curr_player, next_state)
        return

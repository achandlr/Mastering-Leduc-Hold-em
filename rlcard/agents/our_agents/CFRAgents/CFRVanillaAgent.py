from rlcard.agents.our_agents.CFRAgents.CFRBaseAgent import CFRBaseAgent
import numpy as np

class CFRVanillaAgent(CFRBaseAgent):
    """
    Implementation of the Vanilla CFR algorithm.
    
    Inherits from CFRBaseAgent.

    The main characteristics of the Vanilla CFR algorithm:
    - Uses simultaneous updates.
    - Regrets are updated for all players simultaneously.
    - Does not reset negative regrets to 0 after each iteration.
    """
    def __init__(self, env, model_path, perform_logging=False, load_saved_model=True, save_model=True, **kwargs):
        """
        Initialize CFRVanillaAgent.

        Args:
            env: The environment.
            model_path (str): The path to the model.
            perform_logging (bool): Whether to perform logging.
            load_saved_model (bool): Whether to load a saved model.
            save_model (bool): Whether to save the model during training.
            **kwargs: Additional keyword arguments to pass to the base class.
        """
        super().__init__(env, model_path, perform_logging, load_saved_model, save_model, **kwargs)


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
     

    def update_policy(self):
        """Update policy based on regrets using regret matching."""
        for info_set, regrets in self.regrets.items():
            positive_regrets = np.maximum(regrets, 0)
            normalizing_sum = positive_regrets.sum()

            if normalizing_sum > 0:
                self.policy[info_set] = positive_regrets / normalizing_sum  
            else:
                num_actions = self.env.num_actions
                self.policy[info_set] = np.ones(num_actions) / num_actions

            self.cumulative_policy_estimate[info_set] += self.policy[info_set]


    def update_regrets_game_tree(self, probs, player_id):

        if self.env.is_over():
            payoffs = self.env.get_payoffs()
            return payoffs
        
        player_index = self.env.get_player_id()
        # This is the expected utility for the current player in the current state, assuming they follow their current policy.
        state_utility_matrix = np.zeros(self.env.num_players)
        # only stores action utility of player 0, because values are exact opposites
        action_utility_matrix_curr_player = np.zeros(self.env.num_actions)
        curr_player_id = self.env.get_player_id()
        observed_state_tuple, legal_actions = self.observe_state_and_actions(curr_player_id)
        all_action_probs = self.get_action_probs(observed_state_tuple, legal_actions, self.policy)

        for action in legal_actions:
            action_prob = all_action_probs[action]
            recursive_probs = probs.copy()
            recursive_probs[curr_player_id] *= action_prob
            self.env.step(action)
            estimated_utility_matrix = self.update_regrets_game_tree(recursive_probs, player_id)
            self.env.step_back()
            action_utility_matrix_curr_player[action] = estimated_utility_matrix[curr_player_id]
            state_utility_matrix += action_prob * estimated_utility_matrix
        assert state_utility_matrix[0] == -state_utility_matrix[1]

        # only update regrets for the player that is currently being trained
        if curr_player_id == player_id:
            curr_player_prob = probs[curr_player_id]
            # Note: this method is more complicated in over two-player games
            if player_index == 0:
                counter_factual_prob = probs[1]
            else:
                counter_factual_prob = probs[0]
            curr_player_state_utility = state_utility_matrix[curr_player_id]

            # avg policy is approximate Nash equilibrium policy for the current player in the game.
            if observed_state_tuple not in self.cumulative_policy_estimate:
                self.cumulative_policy_estimate[observed_state_tuple] = np.zeros(self.env.num_actions)
            # set regrets for obs if obs not in regrets
            if observed_state_tuple not in self.regrets:
                self.regrets[observed_state_tuple] = np.zeros(self.env.num_actions)
            # This for loop is responsible for updating the regrets and the average policy for the current player at the current observation observed_state_tuple (game state)
            for action in legal_actions:
                hypothetical_action_utility_curr_player = all_action_probs[action]
                regret_for_hypothetical_action = counter_factual_prob * (action_utility_matrix_curr_player[action]
                    - curr_player_state_utility)
                self.regrets[observed_state_tuple][action] += regret_for_hypothetical_action

                '''
                The use of self.training_iteration_index as a weight ensures that later iterations, which have more accurate regret estimates, have a higher influence on the final average policy. We use the same method of increasing weight by training index as cfr_agent to ensure consistency in our evaluation comparison between the two CFR Vanilla agents.
                '''
                self.cumulative_policy_estimate[observed_state_tuple][action] += self.training_iteration_index * curr_player_prob * hypothetical_action_utility_curr_player
            return state_utility_matrix
        else:
            return state_utility_matrix

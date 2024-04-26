from rlcard.agents.our_agents.CFRAgents.CFRBaseAgent import CFRBaseAgent
import numpy as np
import os
import pickle

'''
Explanation of the differences between CFR and CFR+: https://aipokertutorial.com/cfr-advances/
Summarization of Explanation: CFR+ improves upon traditional regret matching by introducing regret matching plus, where any negative regret_sum is reset to zero, ensuring all regret terms remain non-negative. This adjustment allows actions to be re-selected after they have shown utility, preventing them from being disregarded due to accumulated negative regret. 
'''

class CFRPlusAgent(CFRBaseAgent):
    """
    Implementation of CFR+ algorithm.
    
    The main differences compared to vanilla CFR:
    - Use alternating updates instead of simultaneous updates.
    - Reset negative regrets to 0 after each iteration.
    - Use linear averaging for policy updates.
    """

    def __init__(self, env, model_path, perform_logging=False, load_saved_model=True, save_model=True, **kwargs):
        """
        Initialize CFRPlusAgent.

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
     
    
    def update_regrets_game_tree(self, probs, player_id):
        """
        Identical to update_regrets_game_tree method in CFRAgent, except for the resetting of negative regrets to zero.
        """
        if self.env.is_over():
            return self.env.get_payoffs()
        
        current_player = self.env.get_player_id()
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
            if current_player == 0:
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
                regret_for_hypothetical_action = counter_factual_prob * (action_utility_matrix_curr_player[action]
                    - curr_player_state_utility)        
                # Update regrets
                self.regrets[observed_state_tuple][action] += regret_for_hypothetical_action
                # Note: Reset negative regrets to zero (This is what is unique for CFRPlus compared to CFRVanilla)
                self.regrets[observed_state_tuple][action] = max(self.regrets[observed_state_tuple][action], 0)       
                '''
                The use of self.training_iteration_index as a weight ensures that later iterations, which have more accurate regret estimates, have a higher influence on the final average policy. We use the same method of increasing weight by training index as cfr_agent to ensure consistency in our evaluation comparison between the two CFR Vanilla agents.
                '''
                self.cumulative_policy_estimate[observed_state_tuple][action] += self.training_iteration_index * curr_player_prob * all_action_probs[action]   
            return state_utility_matrix
        else:
            return state_utility_matrix

    def save(self):
        super().save()
        # now save cumulative_policy_estimate as pickle file
        with open(os.path.join(self.model_path, 'cumulative_policy_estimate.pkl'), 'wb') as f:
            pickle.dump(self.cumulative_policy_estimate, f)

    def load(self, attempt_load_best=True): 
        super().load()
        if attempt_load_best:
            try:
                # now load cumulative_policy_estimate as pickle file
                with open(os.path.join(self.model_path, 'cumulative_policy_estimate_best'), 'rb') as f:
                    self.cumulative_policy_estimate = pickle.load(f)
            except Exception as e:
                with open(os.path.join(self.model_path, 'cumulative_policy_estimate.pkl'), 'rb') as f:
                    self.cumulative_policy_estimate = pickle.load(f)

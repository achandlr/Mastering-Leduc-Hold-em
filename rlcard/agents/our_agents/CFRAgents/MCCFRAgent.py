from rlcard.agents.our_agents.CFRAgents.CFRBaseAgent import CFRBaseAgent
import numpy as np
from rlcard.utils import remove_illegal
import os
import pickle

'''
MCCFR vs Traditional CFR:
1. MCCFR samples a portion of the game tree on each iteration, while vanilla CFR traverses the entire tree.
2. MCCFR has different sampling schemes (outcome, external, average strategy) that affect convergence speed.
3. MCCFR can be more efficient in large games with many player actions due to faster iterations.
4. Some MCCFR variants (outcome-sampling) enable online regret minimization without full knowledge of the opponent's strategy.

Note on Average Sampling: Average Sampling samples a subset of the player's actions at each information set, whereas Outcome Sampling (OS) samples a single action and External Sampling (ES) samples every action.
'''

class MCCFRAgent(CFRBaseAgent):
    """
    Implementation of Monte Carlo Counterfactual Regret Minimization (MCCFR) algorithm.
    """
    def __init__(self, env, model_path, perform_logging=False, sampling_strategy='external', exploration_rate=0.6, threshold=1, bonus=1, load_saved_model=True, save_model=True, **kwargs):
        """
        Initialize MCCFRAgent.

        Args:
            env: The environment.
            model_path (str): The path to the model.
            perform_logging (bool): Whether to perform logging.
            sampling_strategy (str): The sampling strategy to use ('outcome', 'external', or 'average_strategy').
            exploration_rate (float): The exploration rate for average strategy sampling.
            threshold (float): The threshold for average strategy sampling.
            bonus (float): The bonus for average strategy sampling.
            load_saved_model (bool): Whether to load a saved model.
            save_model (bool): Whether to save the model during training.
            **kwargs: Additional keyword arguments to pass to the base class.
        """
        super().__init__(env, model_path, perform_logging, load_saved_model, save_model, **kwargs)
        self.sampling_strategy = sampling_strategy
        self.exploration_rate = exploration_rate
        self.threshold = threshold
        self.bonus = bonus

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
    
    def get_action_probs(self, info_set, legal_actions, policy):
        """
        Get action probabilities for an information set based on the sampling strategy.
        
        Args:
            info_set (str): The information set.
            legal_actions (list): The legal actions.
            policy (dict): The policy dictionary.
            
        Returns:
            (np.array) The action probabilities.
        """
        sampling_strategy = self.sampling_strategy
        if info_set not in policy:
            probs = np.ones(self.env.num_actions) / self.env.num_actions
            self.policy[info_set] = probs
        else:
            probs = policy[info_set]
        
        if sampling_strategy == 'outcome':
            probs = remove_illegal(probs, legal_actions)
        elif sampling_strategy == 'external':
            probs = remove_illegal(probs, legal_actions)
        elif sampling_strategy == 'average_strategy':
            # BUG: TODO: verify this. it could very easily be wrong.
            if info_set in self.cumulative_policy_estimate:
                average_strategy = self.cumulative_policy_estimate[info_set] / np.sum(self.cumulative_policy_estimate[info_set])
            else:
                average_strategy = np.ones(self.env.num_actions) / self.env.num_actions
            probs = np.array([max(self.exploration_rate, (self.bonus + self.threshold * average_strategy[a]) / (self.bonus + np.sum(average_strategy))) for a in range(self.env.num_actions)])
            probs = remove_illegal(probs, legal_actions)
            # Previous bugged version
            # if info_set in self.cumulative_policy_estimate:
            #     average_strategy = self.cumulative_policy_estimate[info_set] / np.sum(self.cumulative_policy_estimate[info_set])
            # else:
            #     average_strategy = np.ones(self.env.num_actions) / self.env.num_actions
            # probs = np.array([max(self.exploration_rate, (self.bonus + self.threshold * average_strategy[a]) / (self.bonus + np.sum(average_strategy))) for a in range(self.env.num_actions)])
            # probs = remove_illegal(probs, legal_actions)
        
        assert np.all(probs >= 0), "Negative probabilities"
        assert len(legal_actions) == 0 or np.isclose(probs.sum(), 1), "Probabilities do not sum to 1"
        
        return probs
    
    def update_regrets_game_tree(self, probs, player_id):
        """
        Update regrets in the game tree based on the chosen sampling strategy.
        """
        if self.env.is_over():
            return self.env.get_payoffs()
        
        current_player = self.env.get_player_id()
        state_utility_matrix = np.zeros(self.env.num_players)

        action_utility_matrix_curr_player = np.zeros(self.env.num_actions)
        curr_player_id = self.env.get_player_id()

        observed_state_tuple, legal_actions = self.observe_state_and_actions(curr_player_id)

        all_action_probs = self.get_action_probs(observed_state_tuple, legal_actions, self.policy)
        for action in legal_actions:
            action_prob = all_action_probs[action]
            if action_prob > 0:
                recursive_probs = probs.copy()
                recursive_probs[curr_player_id] *= action_prob
                self.env.step(action)
                estimated_utility_matrix = self.update_regrets_game_tree(recursive_probs, player_id)
                self.env.step_back()
                action_utility_matrix_curr_player[action] = estimated_utility_matrix[curr_player_id]
                state_utility_matrix += action_prob * estimated_utility_matrix
        assert state_utility_matrix[0] == -state_utility_matrix[1]

        if curr_player_id == player_id:
            curr_player_prob = probs[curr_player_id]
            if current_player == 0:
                counter_factual_prob = probs[1]
            else:
                counter_factual_prob = probs[0]
            curr_player_state_utility = state_utility_matrix[curr_player_id]

            if observed_state_tuple not in self.cumulative_policy_estimate:
                self.cumulative_policy_estimate[observed_state_tuple] = np.zeros(self.env.num_actions)
            if observed_state_tuple not in self.regrets:
                self.regrets[observed_state_tuple] = np.zeros(self.env.num_actions)

            for action in legal_actions:
                if all_action_probs[action] > 0:
                    regret_for_hypothetical_action = counter_factual_prob * (action_utility_matrix_curr_player[action]
                        - curr_player_state_utility) / all_action_probs[action]
                    self.regrets[observed_state_tuple][action] += regret_for_hypothetical_action
                    self.regrets[observed_state_tuple][action] = max(self.regrets[observed_state_tuple][action], 0)
                    self.cumulative_policy_estimate[observed_state_tuple][action] += self.training_iteration_index * curr_player_prob * all_action_probs[action]

            if self.perform_logging:
                print(f"state utility matrix OUR condition1: {state_utility_matrix}")
            return state_utility_matrix
        else:
            if self.perform_logging:
                print(f"state utility matrix OUR condition2: {state_utility_matrix}")
            return state_utility_matrix

    def save(self):
        super().save()
        # now save cumulative_policy_estimate as pickle file
        with open(os.path.join(self.model_path, 'cumulative_policy_estimate.pkl'), 'wb') as f:
            pickle.dump(self.cumulative_policy_estimate, f)

    def load(self, attempt_load_best=True): 
        super().load()
        # now load cumulative_policy_estimate as pickle file
        if attempt_load_best:
            try:
                with open(os.path.join(self.model_path, 'cumulative_policy_estimate_best'), 'rb') as f:
                    self.cumulative_policy_estimate = pickle.load(f)
            except Exception as e:
                with open(os.path.join(self.model_path, 'cumulative_policy_estimate.pkl'), 'rb') as f:
                    self.cumulative_policy_estimate = pickle.load(f)
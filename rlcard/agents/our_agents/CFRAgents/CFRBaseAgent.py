import numpy as np
from collections import defaultdict
import abc
from rlcard.agents.our_agents.BaseAgent import BaseAgent
from rlcard.utils import remove_illegal

class CFRBaseAgent(BaseAgent, abc.ABC):
    """
    Abstract base class for CFR (Counterfactual Regret Minimization) agents.

    Attributes:
        env (Env): The environment to train the agent on.
        model_path (str): Path to save the trained model.
        regrets (defaultdict): Regret values for each information set and action.
        policy (defaultdict): Current policy for each information set.
        cumulative_policy_estimate (defaultdict): Average policy for each information set over all iterations.
        iteration (int): Current iteration number.
    """

    def __init__(self, env, model_path, perform_logging=False, load_saved_model=True, save_model=True):
        """
        Initialize the CFRBaseAgent.

        Args:
            env (Env): The environment to train the agent on.
            model_path (str): Path to save the trained model.
        """
        super().__init__(env, model_path, perform_logging, load_saved_model, save_model)
        self.regrets = defaultdict(np.array)
        self.policy = defaultdict(list)
        self.cumulative_policy_estimate = defaultdict(np.array)

    @abc.abstractmethod
    def update_regrets_game_tree(self, probs, player_id):
        """
        Update regrets for each information set and action in the game tree.

        Args:
            probs (np.array): The probabilities of reaching the current state.
            player_id (int): The ID of the current player.

        Returns:
            (float) The state utility.
        """
        raise NotImplementedError("Child class must implement this method")

    def get_action_probs(self, info_set, legal_actions, policy):
        """
        Get action probabilities for an information set.

        Args:
            info_set (str): The information set.
            legal_actions (list): The legal actions.
            policy (defaultdict): The current policy for each information set.

        Returns:
            (np.array) The action probabilities.
        """
        if info_set not in policy:
            # Default to uniform distribution over legal actions
            probs = np.ones(self.env.num_actions) / self.env.num_actions
            # This sets current policy because we don't have a policy for this state yet
            self.policy[info_set] = probs
        else:
            probs = policy[info_set]
        # Method available in utils.py that removes illegal actions and renormalizes the probabilities
        probs = remove_illegal(probs, legal_actions)

        assert np.all(probs >= 0), "Negative probabilities"
        assert len(legal_actions) == 0 or np.isclose(probs.sum(), 1), "Probabilities do not sum to 1"

        return probs

    def update_policy(self):
        """
        Update policy based on current regrets.
        """
        for obs, regret in self.regrets.items():
            positive_regret = np.maximum(regret, 0)
            normalizing_sum = np.sum(positive_regret)

            if normalizing_sum > 0:
                self.policy[obs] = positive_regret / normalizing_sum
            else:
                self.policy[obs] = np.ones(self.env.num_actions) / self.env.num_actions

    def train(self):
        """
        Perform one iteration of CFR.

        Traverses the game tree to update regrets for each player and
        information set, then updates strategies based on accumulated regrets.
        """
        self.training_iteration_index += 1

        state_utility_list = []

        # Compute counterfactual regret for each player
        for player_id in range(self.env.num_players):
            self.env.reset()
            probs = np.ones(self.env.num_players)
            state_utility = self.update_regrets_game_tree(probs, player_id)
            state_utility_list.append(state_utility)
        self.update_policy()

    def eval_step(self, state, ranomize_actions=False):
        """
        Predict action based on average policy for a given state.

        Args:
            state (dict): The state to predict an action for.
            ranomize_actions (bool): Whether to randomize actions.

        Returns:
            (int) The predicted action.
            (dict) An empty dictionary (compatibility with other agents).
        """
        state_obs = tuple(state['obs'])
        legal_actions = list(state['legal_actions'].keys())
        action_probabilities = self.get_action_probs(state_obs, legal_actions, policy=self.cumulative_policy_estimate)
        if ranomize_actions:
            chosen_action = np.random.choice(len(action_probabilities), p=action_probabilities)
        else:
            chosen_action = np.argmax(action_probabilities)

        # Note: We keep this similar to existing implementation to ensure compatibility with the state produced by the environment.
        info = {}
        info['probs'] = {state['raw_legal_actions'][i]: float(action_probabilities[list(state['legal_actions'].keys())[i]]) for i in range(len(state['legal_actions']))}
        if self.perform_logging:
            self.logger.info(f"Current output to CFRAgent eval_step is action: {chosen_action}, info: {info}")
        return chosen_action, info

    def save(self):
        """
        Save the model.
        """
        things_to_save = {"policy.pkl": self.policy, "regrets.pkl": self.regrets, "cumulative_policy_estimate.pkl": self.cumulative_policy_estimate, "training_iteration_index.pkl": self.training_iteration_index}
        super().save(things_to_save)

    def load(self):
        """
        Load the model.
        """
        files_to_load = ["policy.pkl", "regrets.pkl", "cumulative_policy_estimate.pkl"] # removed "iteration.pkl"
        super().load(files_to_load)

    def save_only_best(self, reward):
        """
        Save the model only if the reward is the best seen so far.

        Args:
            reward (float): The reward to compare against.
        """
        things_to_save = {"policy.pkl": self.policy, "regrets.pkl": self.regrets, "cumulative_policy_estimate.pkl": self.cumulative_policy_estimate, "training_iteration_index.pkl": self.training_iteration_index}
        super().save_only_best(reward, things_to_save)

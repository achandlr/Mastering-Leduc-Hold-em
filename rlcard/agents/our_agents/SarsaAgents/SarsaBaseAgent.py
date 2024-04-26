from rlcard.agents.our_agents.BaseAgent import BaseAgent


class SarsaBaseAgent(BaseAgent):
    """
    Base class for SARSA agents in RLCard.

    Args:
        config (dict): Configuration dictionary containing agent parameters.

    Attributes:
        gamma (float): Discount factor for future rewards.
        eps_start (float): Initial value of epsilon for epsilon-greedy exploration.
        lowest_epsilon_value (float): Lowest value of epsilon to decay to.
        eps_decay_factor (float): Decay factor for epsilon.
        curr_episode_index (int): Current episode index.
        alpha (float): Learning rate for updating Q-values.

    Methods:
        eval_step(state): Perform an evaluation step for the agent.
        get_epsilon(): Get the current value of epsilon.
        observe_state_and_actions(curr_player_id): Observe the current state and legal actions for a player.
    """

    def __init__(self, config):
        # Extract environment and model path from the config dictionary
        env = config['env']
        model_path = config['model_path']

        # Extract optional parameters from the config dictionary with default values
        perform_logging = config.get('perform_logging', False)
        load_saved_model = config.get('load_saved_model', True)
        save_model = config.get('save_model', True)

        # Call the parent class's __init__ method with the extracted parameters
        super().__init__(env=env, model_path=model_path, perform_logging=perform_logging,
                         load_saved_model=load_saved_model, save_model=save_model)

        # Extract Sarsa-specific parameters from the config dictionary with default values
        self.gamma = config.get('gamma', 0.8)
        self.eps_start = config.get('eps_start', 1)
        self.lowest_epsilon_value = config.get('lowest_epsilon_value', 1e-2)
        self.eps_decay_factor = config.get('eps_decay_factor', 0.999)
        self.curr_episode_index = config.get('curr_episode_index', 0)
        self.alpha = config.get('alpha', 0.001)

    def eval_step(self, state):
        """
        Perform an evaluation step for the agent.

        Args:
            state (object): The current state of the environment.

        Returns:
            tuple: Tuple containing the chosen action and additional information.
        """
        chosen_action = self.select_action(state, use_epsilon_greedy=False)
        info = {}
        return chosen_action, info

    def get_epsilon(self):
        """
        Get the current value of epsilon.

        Returns:
            float: The current value of epsilon.
        """
        epsilon = max(self.eps_start * (self.eps_decay_factor ** self.curr_episode_index), self.lowest_epsilon_value)
        return epsilon
    
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
    
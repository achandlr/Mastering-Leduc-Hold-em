import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from rlcard.utils import remove_illegal
from rlcard.agents.our_agents.CFRAgents.MCCFRAgent import MCCFRAgent
import torch.nn.functional as F
import os 
import pickle
    
class AdvantageNetwork(nn.Module):
    """
    Neural network model for estimating regret in DeepCFR.
    """
    # TODO: Issue might be that this code does not work due to there being two players. Note: is the state representation look the same for each player (they will have different state representations), but still can train from both I think
    def __init__(self, env, neural_network_mode):
        """
        Initialize the AdvantageNetwork.

        Args:
            env (object): The environment object (Leduc Hold'em).
            model_params (dict): Model parameters.
            neural_network_mode (str): The mode of the neural network. Can be 'full' or 'basic'.
        """
        super().__init__()
        self.env = env
        self.neural_network_mode = neural_network_mode
        state_length = env.state_shape[0][0]  # Note: this assumes state shape is a 1D tensor of 36 values
        num_outputs = max_num_actions = len(env.actions)
        assert num_outputs == 4
        if self.neural_network_mode == 'full':
            # Define the full neural network architecture
            INTERMEDIATE_REPRESENTATION_SIZE_1 = 64
            INTERMEDIATE_REPRESENTATION_SIZE_2 = 32
            self.layers = nn.Sequential(
                nn.Linear(state_length, INTERMEDIATE_REPRESENTATION_SIZE_1),
                nn.ReLU(),
                nn.Linear(INTERMEDIATE_REPRESENTATION_SIZE_1,INTERMEDIATE_REPRESENTATION_SIZE_1),
                nn.ReLU(),
                nn.Linear(INTERMEDIATE_REPRESENTATION_SIZE_1, INTERMEDIATE_REPRESENTATION_SIZE_2),
                nn.ReLU(),
                nn.Linear(INTERMEDIATE_REPRESENTATION_SIZE_2, INTERMEDIATE_REPRESENTATION_SIZE_2),
                nn.ReLU(),
                nn.Linear(INTERMEDIATE_REPRESENTATION_SIZE_2, num_outputs)
            )
        elif self.neural_network_mode == 'three_layers':
            # Define the full neural network architecture
            INTERMEDIATE_REPRESENTATION_SIZE_1 = 64
            INTERMEDIATE_REPRESENTATION_SIZE_2 = 32
            self.layers = nn.Sequential(
                nn.Linear(state_length, INTERMEDIATE_REPRESENTATION_SIZE_1),
                nn.ReLU(),
                nn.Linear(INTERMEDIATE_REPRESENTATION_SIZE_1,INTERMEDIATE_REPRESENTATION_SIZE_2),
                nn.ReLU(),
                nn.Linear(INTERMEDIATE_REPRESENTATION_SIZE_2, num_outputs)
            )
        elif self.neural_network_mode == 'basic':
            # Define the basic neural network architecture
            INTERMEDIATE_REPRESENTATION_SIZE = 64
            self.layers = nn.Sequential(
                nn.Linear(state_length, INTERMEDIATE_REPRESENTATION_SIZE),
                nn.ReLU(),
                nn.Linear(INTERMEDIATE_REPRESENTATION_SIZE, num_outputs)
            )
        else:
            raise ValueError(f"Invalid neural_network_mode: {self.neural_network_mode}")
    
    def forward(self, x):
        """
        Forward pass of the neural network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor (counterfactual values).
        """
        return self.layers(x)
    
class DeepCFRAgent(MCCFRAgent):
    """
    Implementation of Deep Counterfactual Regret Minimization (DeepCFR) algorithm for Leduc Hold'em.
    """

    def __init__(self, env, model_path, train_params={}, perform_logging=False, sampling_strategy='external', neural_network_mode='three_layers', load_saved_model=True, save_model=True, **kwargs):
        """
        Initialize DeepCFRAgent.

        Args:
            env: The environment.
            model_path (str): The path to the model.
            train_params (dict): Training parameters for the advantage network.
            perform_logging (bool): Whether to perform logging.
            sampling_strategy (str): The sampling strategy to use ('outcome', 'external', or 'average_strategy').
            neural_network_mode (str): The mode of the neural network ('full' or 'basic').
            load_saved_model (bool): Whether to load a saved model.
            save_model (bool): Whether to save the model during training.
            **kwargs: Additional keyword arguments to pass to the base class.
        """
        super().__init__(env, model_path, perform_logging, sampling_strategy, load_saved_model, save_model, **kwargs)
        self.adv_net = AdvantageNetwork(env, neural_network_mode = neural_network_mode)
        if not train_params: # TODO: confirm this
            train_params = {
                'learning_rate': 0.001,
                'num_epochs': 2,
                'batch_size': 4, # TODO: change these
            }
        self.train_params = train_params
        self.adv_mem = []
        self.neural_network_mode = neural_network_mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train_advantage_network(self):
        # Convert advantage memory to tensors
        states, advantages = zip(*self.adv_mem)
        states = np.array(states)  # Convert list of numpy arrays to a single numpy array
        advantages = np.array(advantages)
        states = torch.tensor(states).float().to(self.device)
        advantages = torch.tensor(advantages).float().to(self.device)
        
        # Create a dataset and data loader for batch training
        dataset = torch.utils.data.TensorDataset(states, advantages)
        
        # Train the advantage network
        optimizer = optim.Adam(self.adv_net.parameters(), lr=self.train_params['learning_rate'])
        criterion = nn.MSELoss()
        
        for _ in range(self.train_params['num_epochs']):
            # By inserting the dataset into the DataLoader, we shuffle the data for each epoch
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.train_params['batch_size'], shuffle=True)

            for batch_states, batch_advantages in data_loader:
                optimizer.zero_grad()
                output = self.adv_net(batch_states)
                loss = criterion(output, batch_advantages)
                loss.backward()
                optimizer.step()
        
        # Clear the advantage memory
        self.adv_mem = []
    
    # This is training for one episode
    def train(self):
        """
        Perform one iteration of CFR.

        Traverses the game tree to update regrets for each player and
        information set, then updates strategies based on accumulated regrets.
        """

        for player_id in range(self.env.num_players):
            self.env.reset()
            probs = np.ones(self.env.num_players)
            self.update_regrets_game_tree(probs, player_id)
            # Train the advantage network in batches
            if len(self.adv_mem) >= self.train_params['batch_size']:
                self.train_advantage_network()
            else:
                self.logger.warning(f"Advantage memory length is less than batch size: {len(self.adv_mem)}")
        self.update_policy()

    def compute_average_strategy(self):
        self.average_strategy = {}
        for info_set, cum_policy_estimate in self.cumulative_policy_estimate.items():
            self.average_strategy[info_set] = cum_policy_estimate / self.cumulative_policy_estimate[info_set].sum()

    def eval_step(self, state, randomize_actions=False):
        """
        Predict action based on the learned advantage network.

        Args:
            state (dict): The state to predict an action for.
            randomize_actions (bool): Whether to sample actions based on probabilities or choose the best action.

        Returns:
            (int) The predicted action.
            (dict) A dictionary containing the action probabilities.
        """
        state_obs = tuple(state['obs'])
        state_obs_tensor = torch.tensor(state_obs).float().to(self.device)
        legal_actions = list(state['legal_actions'].keys())

        # TODO: Confirm this logic. This is likely wrong.
        if state_obs in self.cumulative_policy_estimate:
            probs = self.cumulative_policy_estimate[state_obs]
        else:
            state_obs_tensor = torch.tensor(state_obs).float().to(self.device)
            advantages = self.adv_net(state_obs_tensor).detach().numpy()
            probs = F.softmax(torch.tensor(advantages), dim=0).numpy()
        
        probs = remove_illegal(probs, legal_actions)

        # Use the advantage network to estimate advantages
        # advantages = self.adv_net(state_obs_tensor).detach().numpy()
        # probs = F.softmax(torch.tensor(advantages), dim=0).numpy()
        # probs = remove_illegal(probs, legal_actions)

        if randomize_actions:
            chosen_action = np.random.choice(len(probs), p=probs)
        else:
            chosen_action = np.argmax(probs)
        info = {}
        info['probs'] = {state['raw_legal_actions'][i]: float(probs[list(state['legal_actions'].keys())[i]]) for i in range(len(state['legal_actions']))}
        if self.perform_logging:
            self.logger.info(f"Current output to DeepCFRAgent eval_step is action: {chosen_action}, info: {info}")
        return chosen_action, info
    def update_regrets_game_tree(self, probs, player_id):
        # BUG: self.cumulative_policy_estimate is updated but not used, not even in eval step
        # Same as base update_regrets_game_tree, but adds regrets for each action to adv_mem
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
            # When there are only two players, the counterfactual probability for a player i is simply the probability of the other player taking the actions that led to the current history h. 
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
            regrets_for_each_action_in_state  = np.zeros(self.env.num_actions)
            for action in legal_actions:
                hypothetical_action_utility_curr_player = all_action_probs[action]
                '''
                positive regret = should have done that
                negative regret = should not have done that
                '''
                regret_for_hypothetical_action = counter_factual_prob * (action_utility_matrix_curr_player[action]
                    - curr_player_state_utility)
                regrets_for_each_action_in_state[action] = regret_for_hypothetical_action
                self.regrets[observed_state_tuple][action] += regret_for_hypothetical_action
                self.cumulative_policy_estimate[observed_state_tuple][action] += self.training_iteration_index * curr_player_prob * hypothetical_action_utility_curr_player
            self.adv_mem.append((observed_state_tuple, regrets_for_each_action_in_state))
            return state_utility_matrix
        else:
            return state_utility_matrix

    def save_only_best(self, reward):
        if reward > self.best_reward:
            self.best_reward = reward
        else:
            return
        items_to_save = {"adv_net_best.pkl": self.adv_net,  "training_iteration_index_best.pkl": self.training_iteration_index}
        if not self.save_model:
            return 
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        for file_name, item in items_to_save.items():
            with open(os.path.join(self.model_path, file_name),'wb') as f:
                pickle.dump(item, f)

    def save(self):
        items_to_save = {"adv_net.pkl": self.adv_net,  "training_iteration_index.pkl": self.training_iteration_index}
        if not self.save_model:
            return 
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        for file_name, item in items_to_save.items():
            with open(os.path.join(self.model_path, file_name),'wb') as f:
                pickle.dump(item, f)

    def load(self, attempt_load_best=True):
        files_to_load = ["adv_net.pkl", "training_iteration_index.pkl"]
        if not self.load_saved_model:
            return
        for file_name in files_to_load:
            attribute_name = file_name.split(".")[0]
            if attempt_load_best:
                if file_name.endswith('.pkl'):
                    file_path = os.path.join(self.model_path, file_name[:-4]+"_best.pkl")
                else:
                    file_path = os.path.join(self.model_path, file_name+"_best")
                if os.path.exists(file_path):
                    with open(file_path, 'rb') as file:
                        setattr(self, attribute_name, pickle.load(file))
                    continue
            file_path = os.path.join(self.model_path, file_name)
            if os.path.exists(file_path):
                with open(file_path, 'rb') as file:
                    setattr(self, attribute_name, pickle.load(file))
            else:
                self.logger.info(f"Expected file {file_name} not found in {self.model_path}.")
                raise FileNotFoundError(f"Expected file {file_name} not found in {self.model_path}.")

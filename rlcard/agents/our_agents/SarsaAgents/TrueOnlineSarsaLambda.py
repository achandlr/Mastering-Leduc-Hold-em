import numpy as np
from rlcard.agents.our_agents.SarsaAgents.SarsaBaseAgent import SarsaBaseAgent

class TrueOnlineSarsaLambdaAgent(SarsaBaseAgent):
    def __init__(self, config):
        """
        Initializes the TrueOnlineSarsaLambda agent.

        Args:
            config (dict): A dictionary containing the configuration parameters for the agent.

        Attributes:
            w (numpy.ndarray): The weight matrix for the agent's state-action values.
            lam (float): The eligibility trace decay parameter.

        """
        super().__init__(config)
        self.w = np.zeros((self.num_state_features, self.num_actions))
        self.lam = config.get('lam', 0.9)

    def save(self):
        items_to_save = {"w": self.w}
        super().save(items_to_save)

    def save_only_best(self, reward):
        items_to_save = {"w": self.w}
        super().save_only_best(reward, items_to_save)

    def load(self):
        files_to_load = ["w"]
        super().load(files_to_load)

    def select_action(self, state, use_epsilon_greedy=False):
        # Get the legal actions for the current state
        legal_actions = list(state['legal_actions'].keys())

        # Compute the estimated Q-values for each action using the weight matrix w
        q_values = np.dot(state['obs'], self.w)
        
        if use_epsilon_greedy:
            # Generate a random value between 0 and 1
            val_from_zero_to_one = np.random.rand()
            
            # Get the current epsilon value
            curr_epsilon = self.get_epsilon()
            
            if val_from_zero_to_one < curr_epsilon:
                # Choose a random action with probability epsilon
                chosen_action = np.random.choice(legal_actions)
            else:
                # Choose the action with the highest estimated Q-value
                q_values_legal = q_values[legal_actions]
                chosen_action = legal_actions[np.argmax(q_values_legal)]
        else:
            # Choose the action with the highest estimated Q-value (greedy)
            q_values_legal = q_values[legal_actions]
            chosen_action = legal_actions[np.argmax(q_values_legal)]
        
        return chosen_action

    def train(self):
        self.curr_episode_index += 1
        for player_id in range(self.env.num_players):
            self.env.reset()
            next_state = None
            done = False

            # Initialize eligibility trace for True Online Sarsa(λ)
            z = np.zeros((self.num_state_features, self.num_actions))
            next_action_selected = None
            # TODO: see why this needs to be a self variable
            q_old = 0  # Initialize Q_old to 0

            while not done:
                if next_state is None:
                    state = self.env.get_state(player_id)
                else:
                    state = next_state

                if next_action_selected is None:
                    action_selected = self.select_action(state, use_epsilon_greedy=True)
                else:
                    action_selected = next_action_selected

                # Take action A, observe R, S' (pseudocode line)
                self.env.step(action_selected)
                next_state = self.env.get_state(player_id)
                done = self.env.is_over()

                if done:
                    reward = self.env.get_payoffs()
                    R = reward[player_id]
                else:
                    R = 0

                if not done:
                    # Choose A' ∼ π(·|S') or near greedily from S' using w (pseudocode line)
                    next_action_selected = self.select_action(next_state, use_epsilon_greedy=True)
                else:
                    next_action_selected = None
                    pass # TODO: is this correct? is next_action_selected still used pointing to an invalid variable

                # Compute the feature vectors for the current and next state-action pairs
                x = np.zeros((self.num_state_features, self.num_actions))
                x[:, action_selected] = state['obs']
                x_next = np.zeros((self.num_state_features, self.num_actions))
                if not done:
                    x_next[:, next_action_selected] = next_state['obs']
                else:
                    pass

                Q = np.dot(self.w.flatten(), x.flatten())
                if done:
                    Q_next = 0
                else:
                    Q_next = np.dot(self.w.flatten(), x_next.flatten())
                
                # δ ← R + γQ' - Q_old (pseudocode line)
                delta = R + self.gamma * Q_next - q_old
                
                # z ← γλz + (1 - αγλz^⊤x)x (pseudocode line)
                z_t_x = np.dot(z.flatten(), x.flatten())
                # z_t_x = np.dot(z, x)
                second_part_of_z = (1 - self.alpha * self.gamma * self.lam * z_t_x) * x
                first_par_of_z_update = self.gamma * self.lam * z
                z_update = first_par_of_z_update + second_part_of_z
                z += z_update
                
                # w ← w + α(δ + Q - Q_old)z - α(Q - Q_old)x (pseudocode line)
                self.w += self.alpha * (delta + Q - q_old) * z - self.alpha * (Q - q_old) * x
                
                # Q_old ← Q' (pseudocode line)
                q_old = Q_next

                if done:
                    break

        return

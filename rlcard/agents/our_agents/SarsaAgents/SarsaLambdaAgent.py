import numpy as np
from rlcard.agents.our_agents.SarsaAgents.SarsaBaseAgent import SarsaBaseAgent

class SarsaLambdaAgent(SarsaBaseAgent):
    def __init__(self, config):
        """
        Initialize the SarsaLambdaAgent.

        Args:
            config (dict): Configuration dictionary containing agent parameters.

        Attributes:
            w (numpy.ndarray): Weight matrix of shape (num_state_features, num_actions).
            lam (float): Lambda value for eligibility traces.

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
        legal_actions = list(state['legal_actions'].keys())
        q_values = np.dot(state['obs'], self.w)  # Compute Q-values for all actions using the learned weights w
        val_from_zero_to_one = np.random.rand()
        if use_epsilon_greedy:
            curr_epsilon = self.get_epsilon()
        else:
            curr_epsilon = 0  # This value indicates that we are not using epsilon greedy. Will only pick random action if the state is not in policy
        if val_from_zero_to_one < curr_epsilon:
            chosen_action = np.random.choice(legal_actions)
        else:
            # Exploit: choose the best action based on Q-values
            q_values_legal = q_values[legal_actions]  # Filter Q-values for legal actions only
            chosen_action = legal_actions[np.argmax(q_values_legal)]
        return chosen_action

    def train(self):
        self.curr_episode_index += 1
        for player_id in range(self.env.num_players):
            self.env.reset()
            next_state = None
            done = False
            z = np.zeros((self.num_state_features, self.num_actions))  # Initialize eligibility trace for Sarsa(lambda)
            next_action_selected = None
            while not done:
                if next_state is None:
                    state = self.env.get_state(player_id)
                else:
                    state = next_state
                if next_action_selected is None:
                    action_selected = self.select_action(state, use_epsilon_greedy=True)
                else:
                    action_selected = next_action_selected
                # Take action A, observe R, S'
                self.env.step(action_selected)
                next_state = self.env.get_state(player_id)
                done = self.env.is_over()
                if done:
                    reward = self.env.get_payoffs()
                    R = reward[player_id]
                else:
                    R =  0
                
                # delta = R
                delta = R - np.sum(self.w[:, action_selected] * state['obs'])
                # Loop for i in F(S, A): Update eligibility traces
                for i in range(self.num_state_features):
                    if state['obs'][i] == 1:
                        z[i, action_selected] += 1  # Accumulating traces
                if done:
                    # w ← w + αδz
                    self.w += self.alpha * delta * z
                    # Go to next episode
                    break
                # Choose A' ~ π(·|S') or near greedily ~ q̂(S', ·, w)
                next_action_selected = self.select_action(next_state, use_epsilon_greedy=True) # (ε-greedy), not sure if this is best # TODO: Might be issue with selecting action here and it being different than action_selected in next iteration of while loop
                # Loop for i in F(S', A'): δ ← δ + γwi
                delta += self.gamma * np.sum(self.w[:, next_action_selected] * next_state['obs'])
                # for i in range(self.num_state_features):
                #     if state['obs'][i] == 1: # TODO: should this be state or next state?
                #         delta += self.gamma * self.w[i, A_prime] #TODO confirm this line
                self.w += self.alpha * delta * z
                # z ← γλz
                z *= self.gamma * self.lam
                # S ← S'; A ← A'
                # state = next_state
                # state, A = next_state, A_prime
        return
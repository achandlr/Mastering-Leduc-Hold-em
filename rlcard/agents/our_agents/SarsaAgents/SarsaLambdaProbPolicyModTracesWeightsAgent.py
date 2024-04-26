import numpy as np
from collections import defaultdict
from rlcard.agents.our_agents.SarsaAgents.SarsaBaseAgent import SarsaBaseAgent

'''
The SarsaLambdaProbPolicyModTracesWeightsAgent maintains a separate policy dictionary that maps states to action probabilities, while the SarsaLambdaAgent does not use a separate policy dictionary and instead relies on the learned weights to estimate Q-values directly.

The SarsaLambdaProbPolicyModTracesWeightsAgent updates the eligibility trace using a modified version that includes an additional term (1 - alpha * gamma * lam * np.dot(x, x)) multiplied by the state features, whereas the SarsaLambdaAgent uses the standard accumulating traces approach without this modification.

The SarsaLambdaProbPolicyModTracesWeightsAgent initializes the weights as a 1D array with length equal to the number of actions, while the SarsaLambdaAgent initializes the weights as a 2D array with dimensions (num_state_features, num_actions).
'''

class SarsaLambdaProbPolicyModTracesWeightsAgent(SarsaBaseAgent):
    def __init__(self, config):
        """
        Initialize the SarsaLambdaProbPolicyModTracesWeightsAgent.

        Args:
            config (dict): A dictionary containing the configuration parameters.

        Attributes:
            policy (defaultdict): A dictionary representing the policy.
            w (numpy.ndarray): An array of weights for each action.
            lam (float): The lambda value for eligibility traces.

        """
        super().__init__(config)
        self.policy = defaultdict(list)
        self.w = np.zeros(self.env.num_actions)
        self.lam = config.get('lam', 0.9)


    def update_policy(self, state, action, reward, next_state, done):
        state_tuple = tuple(state['obs'])
        next_state_tuple = tuple(next_state['obs'])

        if state_tuple not in self.policy:
            self.policy[state_tuple] = np.zeros(self.env.num_actions)
        if next_state_tuple not in self.policy:
            self.policy[next_state_tuple] = np.zeros(self.env.num_actions)
        # Note: this update policy code is a modified version of my existing code from project four submision
        # Sarsa(lambda) update
        x = self.policy[state_tuple]
        x_next = self.policy[next_state_tuple]
        q = self.w[action]
        q_next = np.dot(self.w, x_next)
        delta = reward + self.gamma * q_next * (1 - done) - q
        z = self.gamma * self.lam * x + (1 - self.alpha * self.gamma * self.lam * np.dot(x, x)) * x
        self.w += self.alpha * (delta + q - np.dot(self.w, x)) * z - self.alpha * (q - np.dot(self.w, x)) * x
        return

    def save(self):
        items_to_save = {"policy": self.policy, "w": self.w}
        super().save(items_to_save)

    def save_only_best(self, reward):
        items_to_save = {"policy": self.policy, "w": self.w}
        super().save_only_best(reward, items_to_save)

    def load(self):
        files_to_load = ["policy", "w"]
        super().load(files_to_load)

    def select_action(self, state, use_epsilon_greedy=False):
        state_obs = tuple(state['obs'])
        legal_actions = list(state['legal_actions'].keys())
        val_from_zero_to_one = np.random.rand()
        if use_epsilon_greedy:
            curr_epsilon = self.get_epsilon()
        else:
            curr_epsilon = 0  # This value indicates that we are not using epsilon greedy. Will only pick random action if the state is not in policy
        if val_from_zero_to_one < curr_epsilon or state_obs not in self.policy or len(self.policy[state_obs]) == 0:
            chosen_action = np.random.choice(legal_actions)
        else:
            # Make all illegal actions have -inf so that they are not selected
            masked_arr = np.full(self.policy[state_obs].shape, -np.inf)
            masked_arr[legal_actions] = self.policy[state_obs][legal_actions]
            chosen_action = np.argmax(masked_arr)
        return chosen_action

    def train(self):
        self.curr_episode_index += 1
        for player_id in range(self.env.num_players):
            self.env.reset()
            next_state = None
            done = False
            while not done:
                if next_state is None:
                    state = self.env.get_state(player_id)
                else:
                    state = next_state
                # select action with epsilon greedy while training
                action = self.select_action(state, use_epsilon_greedy=True)
                self.env.step(action)
                done = self.env.is_over()
                if done:
                    reward = self.env.get_payoffs()[player_id]
                else:
                    reward = 0
                next_state = self.env.get_state(player_id)
                self.update_policy(state, action, reward, next_state, done)
        return
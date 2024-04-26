# Note: This class does not inherit BaseAgent, because it is so trivial that we do not need access to any of the methods in BaseAgent.
class CheckFoldAgent(object):
    ''' 
    An agent that always checks when check is an option and folds otherwise.
    '''

    def __init__(self, num_actions):
        ''' Initialize the CheckFoldAgent.

        Args:
            num_actions (int): The size of the ouput action space
        '''
        self.use_raw = False
        self.num_actions = num_actions

    @staticmethod
    def step(state):
        ''' Predict the action given the curent state in gerenerating training data.

        Args:
            state (dict): An dictionary that represents the current state

        Returns:
            action (int): The action predicted (randomly chosen) by the random agent
        '''
        legal_actions_as_ints_ = state['legal_actions']
        legal_actions_as_strings = state['raw_legal_actions']
        if 'check' in legal_actions_as_strings:
            check_index = legal_actions_as_strings.index('check')
            return legal_actions_as_ints_[check_index]
        elif 'fold' in legal_actions_as_strings:
            fold_index = legal_actions_as_strings.index('fold')
            return legal_actions_as_ints_[fold_index]
        else:
            raise ValueError("No check or fold action available")


    def eval_step(self, state):
        '''
        Predict the action given the current state for evaluation.

        Args:
            state (dict): A dictionary that represents the current state

        Returns:
            action (int): The action predicted by the CheckFoldAgent
            info (dict): A dictionary containing action probabilities
        '''
        probs = [0 for _ in range(self.num_actions)]
        if 'check' in state['raw_legal_actions']:
            action = state['raw_legal_actions'].index('check')
            probs[action] = 1
        elif 'fold' in state['raw_legal_actions']:
            action = state['raw_legal_actions'].index('fold')
            probs[action] = 1
        else:
            raise ValueError("No check or fold action available")

        info = {}
        info['probs'] = {action_name: probs[state['raw_legal_actions'].index(action_name)] if action_name in state['raw_legal_actions'] else 0 for action_name in state['raw_legal_actions']}
        assert sum(info['probs'].values()) == 1
        assert sum(probs) == 1
        return action, info





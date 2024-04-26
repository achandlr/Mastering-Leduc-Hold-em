# Note: This class does not inherit BaseAgent, because it is so trivial that we do not need access to any of the methods in BaseAgent.
class RaiseCallAgent(object):
    ''' An agent that always raises when raise is an option and calls otherwise.
    '''

    def __init__(self, num_actions):
        '''
        Initialize the RaiseCallAgent.

        Args:
            num_actions (int): The size of the output action space
        '''
        self.use_raw = False
        self.num_actions = num_actions

    def step(self, state):
        '''
        Predict the action given the current state in generating training data.

        Args:
            state (dict): A dictionary that represents the current state

        Returns:
            action (int): The action predicted by the RaiseCallAgent
        '''
        if 'raise' in state['raw_legal_actions']:
            return state['raw_legal_actions'].index('raise')
        elif 'call' in state['raw_legal_actions']:
            return state['raw_legal_actions'].index('call')
        else:
            raise ValueError("No raise or call action available")


    def eval_step(self, state):
        '''
        Predict the action given the current state for evaluation.

        Args:
            state (dict): A dictionary that represents the current state

        Returns:
            action (int): The action predicted by the RaiseCallAgent
            info (dict): A dictionary containing action probabilities
        '''
        probs = [0 for _ in range(self.num_actions)]
        if 'raise' in state['raw_legal_actions']:
            action = state['raw_legal_actions'].index('raise')
            probs[action] = 1
        elif 'call' in state['raw_legal_actions']:
            action = state['raw_legal_actions'].index('call')
            probs[action] = 1
        else:
            raise ValueError("No raise or call action available")
        assert sum(probs) == 1
        info = {}
        info['probs'] = {action_name: probs[state['raw_legal_actions'].index(action_name)] if action_name in state['raw_legal_actions'] else 0 for action_name in state['raw_legal_actions']}
        # info['probs'] = {state['raw_legal_actions'][i]: probs[list(state['legal_actions'].keys())[i]] for i in range(len(state['legal_actions']))}
        assert sum(info['probs'].values()) == 1
        return action, info

 
import json
import os
import numpy as np
from collections import OrderedDict

import rlcard
from rlcard.envs import Env
from rlcard.games.leducholdem import Game
from rlcard.utils import *

from datetime import datetime
import logging

DEFAULT_GAME_CONFIG = {
        'game_num_players': 2,
        }

class LeducholdemEnv(Env):
    ''' Leduc Hold'em Environment
    '''

    def __init__(self, config, perform_logging = True):
        ''' Initialize the Limitholdem environment
        '''
        self.name = 'leduc-holdem' 
        self.default_game_config = DEFAULT_GAME_CONFIG
        self.game = Game()
        super().__init__(config)
        self.actions = ['call', 'raise', 'fold', 'check']
        self.state_shape = [[36] for _ in range(self.num_players)]
        self.action_shape = [None for _ in range(self.num_players)]

        with open(os.path.join(rlcard.__path__[0], 'games/leducholdem/card2index.json'), 'r') as file:
            self.card2index = json.load(file)

        self.perform_logging = perform_logging

        if perform_logging:
            # Setup logger
            current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            log_dir = f"logs/leduc_holdem/{current_time}"
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            log_file = os.path.join(log_dir, "leducholdem_env.log")
            
            logging.basicConfig(filename=log_file, level=logging.INFO)
            self.logger = logging.getLogger('LeducholdemEnv')
            self.logger.info("LeducholdemEnv logger initialized.")

            self.logger.info(f"Required input to LeducholdemEnv: {config}")


    def _get_legal_actions(self):
        ''' Get all leagal actions

        Returns:
            encoded_action_list (list): return encoded legal action list (from str to int)
        '''
        legal_actions = self.game.get_legal_actions()
        if self.perform_logging:
            self.logger.info(f"Ouptut to _get_legal_actions: legal_actions: {legal_actions}")
        return legal_actions

    def _extract_state(self, state):
        self.logger.info(f"Required input to _extract_state: {state}")
        ''' Extract the state representation from state dictionary for agent

        Note: Currently the use the hand cards and the public cards. TODO: encode the states

        Args:
            state (dict): Original state from the game

        Returns:
            observation (list): combine the player's score and dealer's observable score for observation
        '''
        if self.perform_logging: 
            self.logger.info(f"Input to _extract_state: state:{state}")
        extracted_state = {}

        legal_actions = OrderedDict({self.actions.index(a): None for a in state['legal_actions']})
        extracted_state['legal_actions'] = legal_actions

        public_card = state['public_card']
        hand = state['hand']
        obs = np.zeros(36)
        obs[self.card2index[hand]] = 1
        if public_card:
            obs[self.card2index[public_card]+3] = 1
        obs[state['my_chips']+6] = 1
        obs[sum(state['all_chips'])-state['my_chips']+21] = 1
        extracted_state['obs'] = obs

        extracted_state['raw_obs'] = state
        extracted_state['raw_legal_actions'] = [a for a in state['legal_actions']]
        extracted_state['action_record'] = self.action_recorder
        if self.perform_logging:
            self.logger.info(f"Output to _extract_state state: {extracted_state}")
        return extracted_state

    def get_payoffs(self):
        ''' Get the payoff of a game

        Returns:
           payoffs (list): list of payoffs
        '''
        payoffs = self.game.get_payoffs()
        if self.perform_logging:
            self.logger.info(f"Required output to get_payoffs: {payoffs}")
        return payoffs

    def _decode_action(self, action_id):
        ''' Decode the action for applying to the game

        Args:
            action id (int): action id

        Returns:
            action (str): action for the game
        '''
        if self.perform_logging:
            self.logger.info(f"Required input to _decode_action: {action_id}")
        list_acceptable_actions = ["call", "raise", "fold", "check"]
        if self.perform_logging:
            self.logger.info(f"Required output to _decode_action will be one of the following strings: {list_acceptable_actions}")
        legal_actions = self.game.get_legal_actions()
        if self.actions[action_id] not in legal_actions:
            if 'check' in legal_actions:
                return 'check'
            else:
                return 'fold'
        return self.actions[action_id]

    def get_perfect_information(self):
        ''' Get the perfect information of the current state

        Returns:
            (dict): A dictionary of all the perfect information of the current state
        '''
        
        state = {}
        state['chips'] = [self.game.players[i].in_chips for i in range(self.num_players)]
        state['public_card'] = self.game.public_card.get_index() if self.game.public_card else None
        # First elem is if space or heart, second is the rank
        state['hand_cards'] = [self.game.players[i].hand.get_index() for i in range(self.num_players)]
        state['current_round'] = self.game.round_counter
        state['current_player'] = self.game.game_pointer
        state['legal_actions'] = self.game.get_legal_actions()
        if self.perform_logging:
            self.logger.info(f"Required output to get_perfect_information: {state}")
        return state

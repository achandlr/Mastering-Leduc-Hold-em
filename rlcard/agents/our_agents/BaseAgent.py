import numpy as np
import os
import logging
from datetime import datetime
import abc
import pickle

class BaseAgent(abc.ABC):
    """
    Abstract base class for all agents.
    Attributes:
        env (Env): The environment to train the agent on.
        model_path (str): Path to save the trained model.
        iteration (int): Current iteration number.
    """

    def __init__(self, env, model_path, perform_logging=False, load_saved_model=True, save_model=True, **kwargs):
        """
        Initialize the CFRBaseAgent.

        Args:
            env (Env): The environment to train the agent on.
            model_path (str): Path to save the trained model.
        """

        # This is needed to interact with the environment
        self.use_raw = False 
        self.env = env
        self.load_saved_model = load_saved_model
        self.save_model = save_model
        self.model_path = model_path
        self.perform_logging = perform_logging
        self.best_reward = -np.inf
        self.training_iteration_index = 0
        self.num_actions = 4
        self.num_state_features = 36
        self.__dict__.update(kwargs)

        if perform_logging:
            # Setup logger
            current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            model_name = model_path.split("/")[-1]
            log_dir = f"logs/model_name/"
            log_name = f"{model_name}_{current_time}.log"
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            log_file = os.path.join(log_dir, log_name)
            
            logging.basicConfig(filename=log_file, level=logging.INFO)
            self.logger = logging.getLogger('CFRAgent')
            self.logger.info(f"{log_name} logger initialized.")

    def select_action(self, state):
        raise NotImplementedError("Child class must implement this method")

    def train(self):
        raise NotImplementedError("Child class must implement this method")
        
    def save_only_best(self, reward, items_to_save):
        if self.save_model and reward > self.best_reward:
            self.best_reward = reward
        else:
            return 
        for file_name, item in items_to_save.items():
            with open(os.path.join(self.model_path, file_name+"_best"),'wb') as f:
                pickle.dump(item, f)

    @abc.abstractmethod
    def eval_step(self, state):
        """
        Predict action based on average policy for a given state.

        Args:
            state (dict): The state to predict an action for.

        Returns:
            (int) The predicted action.
            (dict) An empty dictionary (compatibility with other agents).
        """
        raise NotImplementedError("Child class must implement this method")

    def save(self, items_to_save):
        ''' Save model
        '''
        if not self.save_model:
            return 
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        for file_name, item in items_to_save.items():
            with open(os.path.join(self.model_path, file_name),'wb') as f:
                pickle.dump(item, f)


    def load(self, files_to_load, try_load_best = True):
        ''' Load model
        '''
        if not self.load_saved_model:
            return
        for file_name in files_to_load:
            attribute_name = file_name.split(".")[0]
            if try_load_best:
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
                # self.logger.info(f"Expected file {file_name} not found in {self.model_path}.")
                raise FileNotFoundError(f"Expected file {file_name} not found in {self.model_path}.")
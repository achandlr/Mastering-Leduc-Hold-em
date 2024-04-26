import numpy as np
from rlcard.games.base import Card
import os
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import json
from matplotlib import ticker

def set_seed(seed):
    if seed is not None:
        import subprocess
        import sys

        reqs = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])
        installed_packages = [r.decode().split('==')[0] for r in reqs.split()]
        if 'torch' in installed_packages:
            import torch
            torch.backends.cudnn.deterministic = True
            torch.manual_seed(seed)
        np.random.seed(seed)
        import random
        random.seed(seed)


def get_folders_path(base_folder, num_levels_desired):
    """
    Retrieves the paths of folders that are a specified number of levels down from a base folder.

    Args:
        base_folder (str): The path to the base folder.
        num_levels_desired (int): The number of levels down from the base folder to retrieve folders from.

    Returns:
        list: A list of strings representing the paths of the desired folders.
    """
    desired_paths = []

    def traverse_folders(current_folder, current_level):
        """
        Recursive helper function to traverse the folder structure and retrieve desired folder paths.

        Args:
            current_folder (str): The path to the current folder being traversed.
            current_level (int): The current level of depth in the folder structure.
        """
        if current_level == num_levels_desired:
            # If the current level matches the desired number of levels, add the current folder path to the list
            desired_paths.append(current_folder)
        else:
            # If the current level is less than the desired number of levels, continue traversing subfolders
            for subfolder in os.listdir(current_folder):
                subfolder_path = os.path.join(current_folder, subfolder)
                if os.path.isdir(subfolder_path):
                    traverse_folders(subfolder_path, current_level + 1)

    # Start traversing the folder structure from the base folder
    traverse_folders(base_folder, 0)

    return desired_paths

def get_evaluation_iterations(num_episodes):
    """
    Set the evaluation iterations based on the number of episodes.

    Args:
        args: The arguments object containing the number of episodes.

    Returns:
        None. The evaluation iterations are directly modified in the args object.
    """
    # Define evaluation iterations for different ranges
    every_one_initial = list(range(0, 11, 1))
    every_five_initial = list(range(0, 51, 5))
    every_ten_initial = list(range(0, 101, 10))
    every_twenty_initial = list(range(100, 500 + 1, 25))
    # every_fifty_initial = list(range(100, 1000+1, 50))
    every_hundred_initial = list(range(100, 2_000+1, 100))
    every_five_hundred = list(range(0, 10_000 + 1, 1_000))
    every_10_000 = list(range(0, 100_000_000 + 1, 10_000))

    evaluation_iterations = []
    # Extend the evaluation iterations with the defined ranges
    evaluation_iterations.extend(every_five_initial)
    evaluation_iterations.extend(every_one_initial)
    evaluation_iterations.extend(every_ten_initial)
    evaluation_iterations.extend(every_twenty_initial)
    # evaluation_iterations.extend(every_fifty_initial)
    evaluation_iterations.extend(every_hundred_initial)
    evaluation_iterations.extend(every_five_hundred)
    evaluation_iterations.extend(every_10_000)

    # Remove duplicate evaluation iterations
    evaluation_iterations = set(evaluation_iterations)
    # Remove evaluation iterations that exceed the number of episodes
    evaluation_iterations = [i for i in evaluation_iterations if i <= num_episodes]
    return evaluation_iterations

def get_device():
    import torch
    if torch.backends.mps.is_available():
        device = torch.device("mps:0")
        print("--> Running on the GPU")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("--> Running on the GPU")
    else:
        device = torch.device("cpu")
        print("--> Running on the CPU")

    return device    

def init_standard_deck():
    ''' Initialize a standard deck of 52 cards

    Returns:
        (list): A list of Card object
    '''
    suit_list = ['S', 'H', 'D', 'C']
    rank_list = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']
    res = [Card(suit, rank) for suit in suit_list for rank in rank_list]
    return res

def init_54_deck():
    ''' Initialize a standard deck of 52 cards, BJ and RJ

    Returns:
        (list): Alist of Card object
    '''
    suit_list = ['S', 'H', 'D', 'C']
    rank_list = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']
    res = [Card(suit, rank) for suit in suit_list for rank in rank_list]
    res.append(Card('BJ', ''))
    res.append(Card('RJ', ''))
    return res

def rank2int(rank):
    ''' Get the coresponding number of a rank.

    Args:
        rank(str): rank stored in Card object

    Returns:
        (int): the number corresponding to the rank

    Note:
        1. If the input rank is an empty string, the function will return -1.
        2. If the input rank is not valid, the function will return None.
    '''
    if rank == '':
        return -1
    elif rank.isdigit():
        if int(rank) >= 2 and int(rank) <= 10:
            return int(rank)
        else:
            return None
    elif rank == 'A':
        return 14
    elif rank == 'T':
        return 10
    elif rank == 'J':
        return 11
    elif rank == 'Q':
        return 12
    elif rank == 'K':
        return 13
    return None

def elegent_form(card):
    ''' Get a elegent form of a card string

    Args:
        card (string): A card string

    Returns:
        elegent_card (string): A nice form of card
    '''
    suits = {'S': '♠', 'H': '♥', 'D': '♦', 'C': '♣','s': '♠', 'h': '♥', 'd': '♦', 'c': '♣' }
    rank = '10' if card[1] == 'T' else card[1]

    return suits[card[0]] + rank

def print_card(cards):
    ''' Nicely print a card or list of cards

    Args:
        card (string or list): The card(s) to be printed
    '''
    if cards is None:
        cards = [None]
    if isinstance(cards, str):
        cards = [cards]

    lines = [[] for _ in range(9)]

    for card in cards:
        if card is None:
            lines[0].append('┌─────────┐')
            lines[1].append('│░░░░░░░░░│')
            lines[2].append('│░░░░░░░░░│')
            lines[3].append('│░░░░░░░░░│')
            lines[4].append('│░░░░░░░░░│')
            lines[5].append('│░░░░░░░░░│')
            lines[6].append('│░░░░░░░░░│')
            lines[7].append('│░░░░░░░░░│')
            lines[8].append('└─────────┘')
        else:
            if isinstance(card, Card):
                elegent_card = elegent_form(card.suit + card.rank)
            else:
                elegent_card = elegent_form(card)
            suit = elegent_card[0]
            rank = elegent_card[1]
            if len(elegent_card) == 3:
                space = elegent_card[2]
            else:
                space = ' '

            lines[0].append('┌─────────┐')
            lines[1].append('│{}{}       │'.format(rank, space))
            lines[2].append('│         │')
            lines[3].append('│         │')
            lines[4].append('│    {}    │'.format(suit))
            lines[5].append('│         │')
            lines[6].append('│         │')
            lines[7].append('│       {}{}│'.format(space, rank))
            lines[8].append('└─────────┘')

    for line in lines:
        print ('   '.join(line))

def reorganize(trajectories, payoffs):
    ''' Reorganize the trajectory to make it RL friendly

    Args:
        trajectory (list): A list of trajectories
        payoffs (list): A list of payoffs for the players. Each entry corresponds to one player

    Returns:
        (list): A new trajectories that can be fed into RL algorithms.

    '''
    num_players = len(trajectories)
    new_trajectories = [[] for _ in range(num_players)]

    for player in range(num_players):
        for i in range(0, len(trajectories[player])-2, 2):
            if i ==len(trajectories[player])-3:
                reward = payoffs[player]
                done =True
            else:
                reward, done = 0, False
            transition = trajectories[player][i:i+3].copy()
            transition.insert(2, reward)
            transition.append(done)

            new_trajectories[player].append(transition)
    return new_trajectories

def remove_illegal(action_probs, legal_actions):
    ''' Remove illegal actions and normalize the
        probability vector

    Args:
        action_probs (numpy.array): A 1 dimention numpy array.
        legal_actions (list): A list of indices of legal actions.

    Returns:
        probd (numpy.array): A normalized vector without legal actions.
    '''
    probs = np.zeros(action_probs.shape[0])
    probs[legal_actions] = action_probs[legal_actions]
    if np.sum(probs) == 0:
        probs[legal_actions] = 1 / len(legal_actions)
    else:
        probs /= sum(probs)
    return probs

def tournament(env, num, return_rewards_each_hand = False):
    ''' Evaluate he performance of the agents in the environment

    Args:
        env (Env class): The environment to be evaluated.
        num (int): The number of games to play.

    Returns:
        A list of avrage payoffs for each player
    '''
    payoffs = [0 for _ in range(env.num_players)]
    counter = 0
    if return_rewards_each_hand:
        rewards_each_hand = []
    while counter < num:
        _, _payoffs = env.run(is_training=False)
        if isinstance(_payoffs, list):
            if return_rewards_each_hand:
                rewards_each_hand.append(_payoffs[0])
            for _p in _payoffs:
                for i, _ in enumerate(payoffs):
                    payoffs[i] += _p[i]
                counter += 1
        else:
            if return_rewards_each_hand:
                rewards_each_hand.append(_payoffs[0])
            for i, _ in enumerate(payoffs):
                payoffs[i] += _payoffs[i]
            counter += 1
    for i, _ in enumerate(payoffs):
        payoffs[i] /= counter
    if return_rewards_each_hand:
        # assert 0 not in rewards_each_hand, "There is a 0 in rewards_each_hand"
        return payoffs, rewards_each_hand
    else:
        return payoffs




def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def get_config_path(config_type, config_name):
    return os.path.join('configs', config_type, f'{config_name}.json')

import colorsys
import matplotlib.pyplot as plt
import numpy as np

def get_contrasting_colors(n):
    colors = []
    for i in range(n):
        # Calculate the hue value evenly spaced around the color wheel
        hue = i / n
        # Set saturation and lightness to 1 to get the most vibrant colors
        saturation = 1
        lightness = 0.5
        # Convert HSL color to RGB
        r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
        # Convert RGB from 0-1 range to 0-255 range and format to hex
        hex_color = '#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255))
        colors.append(hex_color)
    return colors



name_mapping = {
    "cfr": "CFR Vanilla (Theirs)",
    "cfr_vanilla": "CFR Vanilla",
    "mccfr": "MCCFR",
    "cfr_plus": "CFR+",
    "deep_cfr": "Deep CFR",
    "mccfr_outcome": "MCCFR-Outcome",
    "mccfr_external": "MCCFR-External",
    "mccfr_average": "MCCFR-Avg",
    "sarsa" : "SARSA",
    "expected_sarsa" : "Expected SARSA",
    "nstep_sarsa" : "n-step SARSA",
    "sarsa_lambda" : "SARSA Lambda",
    "sarsa_lambda_prob_policy_mod_traces_weights" : "SARSA Lambda Variant",
    "true_online_sarsa_lambda" : "True Online SARSA Lambda",
    "deep_true_online_sarsa_lambda" : "Deep True Online SARSA Lambda",
    "mc_control_batch_updates" : "MC Control Batch Updates",
    "n_step_sarsa_variant" : "n-step SARSA Variant",
    "q_learning" : "Q-Learning",
    "reinforce" : "REINFORCE"
}

['deep_cfr', 'sarsa', 'expected_sarsa', 'nstep_sarsa', 'sarsa_lambda', 'sarsa_lambda_prob_policy_mod_traces_weights', 'true_online_sarsa_lambda', 'deep_true_online_sarsa_lambda', 'mc_control_batch_updates', 'q_learning', 'reinforce']


def plot_curves_numbered_lines(agent_to_csv_path, x_axis='episode', fig_name="all_agents.png"):
    '''
    Read data from csv files and plot the results for all agents
    '''
    sns.set_style("darkgrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    needed_num_colors = len(agent_to_csv_path)
    colors = get_contrasting_colors(needed_num_colors)
    agent_to_color = dict(zip(agent_to_csv_path.keys(), colors))
    for i, (agent, csv_path) in enumerate(agent_to_csv_path.items(), start=1):
        with open(csv_path) as csvfile:
            reader = csv.DictReader(csvfile)
            data = {'x': [], 'y': []}
            for row in reader:
                if x_axis == 'episode':
                    data['x'].append(int(row['episode']))
                    x_label = 'Episodes'
                    plot_title = 'Average Reward over Episodes'
                elif x_axis == 'training_time':
                    data['x'].append(float(row['training_time']))
                    x_label = 'Training Time'
                    plot_title = 'Average Reward over Time'
                else:
                    raise ValueError('x_axis should be episode or training_time')
                reward = float(row['reward'])
                data['y'].append(reward)

            special_name = name_mapping.get(agent, agent)
            color = agent_to_color[agent]

            line = sns.lineplot(x='x', y='y', data=data, ax=ax, label=f"{i}. {special_name}", linewidth=2, color=color)

            # Add the line number at the beginning of each line
            x_pos = data['x'][0]
            y_pos = data['y'][0]
            ax.text(x_pos, y_pos, str(i), fontsize=12, color=color, ha='right', va='center')

    ax.legend(fontsize=12, loc='lower right', title='Models')
    ax.set_xlabel(x_label, fontsize=18)
    ax.set_ylabel('Average Reward', fontsize=18)
    ax.set_title(plot_title, fontsize=18)

    plt.tight_layout()

    save_path = f"experiments/{fig_name}"
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    fig.savefig(save_path, dpi=300)
    plt.show()
    return


def plot_curves_line_types(agent_to_csv_path, x_axis='episode', fig_name="all_agents.png"):
    sns.set_style('darkgrid')
    plt.figure(figsize=(8, 6))
    ax = plt.subplot()
    
    # Define line styles for each model type
    line_styles = {
        'cfr': '--',  # Dashed line for CFR models
        'sarsa': '-',  # Solid line for Sarsa models
        'other': ':'  # Dotted line for other models
    }
    
    needed_num_colors = len(agent_to_csv_path)
    colors = get_contrasting_colors(needed_num_colors)
    agent_to_color = dict(zip(agent_to_csv_path.keys(), colors))
    
    for agent, csv_path in agent_to_csv_path.items():
        with open(csv_path) as csvfile:
            reader = csv.DictReader(csvfile)
            data = {'x': [], 'y': []}
            for row in reader:
                if x_axis == 'episode':
                    data['x'].append(int(row['episode']))
                    x_label = 'Episodes'
                    plot_title = 'Average Reward over Episodes'
                elif x_axis == 'training_time':
                    data['x'].append(float(row['training_time']))
                    x_label = 'Training Time'
                    plot_title = 'Average Reward over Time'
                else:
                    raise ValueError('x_axis should be episode or training_time')
                reward = float(row['reward'])
                data['y'].append(reward)
        
        # Determine the line style based on the model type
        line_style = None
        for model_type, style in line_styles.items():
            if model_type in agent.lower():
                line_style = style
                break
        if line_style is None:
            line_style = line_styles['other']
        
        special_name = name_mapping.get(agent, agent)
        color = agent_to_color[agent]
        sns.lineplot(x='x', y='y', data=data, ax=ax, label=special_name, linewidth=2, color=color, linestyle=line_style)
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_major_formatter(ticker.EngFormatter())
    
    plt.title(plot_title, fontsize=20)
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel('Reward', fontsize=14)
    
    # Create a custom legend
    legend_handles = [
        plt.Line2D([], [], linestyle=line_styles['cfr'], color='black', label='CFR Models'),
        plt.Line2D([], [], linestyle=line_styles['sarsa'], color='black', label='Sarsa Models'),
        plt.Line2D([], [], linestyle=line_styles['other'], color='black', label='Other Models')
    ]
    for agent in agent_to_csv_path.keys():
        legend_handles.append(plt.Line2D([], [], linestyle='-', color=agent_to_color[agent], label=name_mapping.get(agent, agent)))
    
    ax.legend(handles=legend_handles, fontsize=12, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.show()
    return

def plot_curves_variant_grouped(agent_to_csv_path, x_axis='episode', fig_name="all_agents.png"):
    '''
    Read data from csv files and plot the results for all agents
    '''
    legend_style = 'grouped' if len(agent_to_csv_path) > 8 else legend_style
    sns.set_style("darkgrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    needed_num_colors = len(agent_to_csv_path)
    colors = get_contrasting_colors(needed_num_colors)
    agent_to_color = dict(zip(agent_to_csv_path.keys(), colors))

    line_styles = {'cfr': '--', 'sarsa': '-', 'other': ':'}
    line_numbers = {}

    for agent, csv_path in agent_to_csv_path.items():
        with open(csv_path) as csvfile:
            reader = csv.DictReader(csvfile)
            data = {'x': [], 'y': []}
            for row in reader:
                if x_axis == 'episode':
                    data['x'].append(int(row['episode']))
                    x_label = 'Episodes'
                    plot_title = 'Average Reward over Episodes'
                elif x_axis == 'training_time':
                    data['x'].append(float(row['training_time']))
                    x_label = 'Training Time'
                    plot_title = 'Average Reward over Time'
                else:
                    raise ValueError('x_axis should be episode or training_time')
                reward = float(row['reward'])
                data['y'].append(reward)

            special_name = name_mapping.get(agent, agent)
            color = agent_to_color[agent]

            if 'cfr' in agent.lower():
                line_style = line_styles['cfr']
                agent_type = 'CFR'
            elif 'sarsa' in agent.lower():
                line_style = line_styles['sarsa']
                agent_type = 'Sarsa'
            else:
                line_style = line_styles['other']
                agent_type = 'Other'

            line_number = line_numbers.get(agent_type, 1)
            line_numbers[agent_type] = line_number + 1

            sns.lineplot(x='x', y='y', data=data, ax=ax, label=f"{special_name} ({line_number})", linewidth=2, color=color, linestyle=line_style)

    if legend_style == 'grouped':
        cfr_handles, sarsa_handles, other_handles = [], [], []
        for handle in ax.get_legend_handles_labels()[0]:
            if handle.get_label().startswith('CFR'):
                cfr_handles.append(handle)
            elif handle.get_label().startswith('Sarsa'):
                sarsa_handles.append(handle)
            else:
                other_handles.append(handle)

        legend_handles = [*cfr_handles, *sarsa_handles, *other_handles]
        legend_labels = [handle.get_label() for handle in legend_handles]
        ax.legend(legend_handles, legend_labels, fontsize=12, loc='lower right', ncol=3, title='Models')
    else:
        ax.legend(fontsize=12, loc='lower right')

    ax.set_xlabel(x_label, fontsize=18)
    ax.set_ylabel('Average Reward', fontsize=18)
    ax.set_title(plot_title, fontsize=18)

    plt.tight_layout()

    save_path = f"experiments/{fig_name}"
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    fig.savefig(save_path, dpi=300)
    plt.show()
    return

def plot_curves(agent_to_csv_path, x_axis='episode', fig_name = "all_agents.png"):
    '''
    Read data from csv files and plot the results for all agents
    '''
    sns.set_style("darkgrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    # BUG: missin gblue and brown (cfr theirs and MCCFR outcome)
    needed_num_colors = len(agent_to_csv_path)
    colors = get_contrasting_colors(needed_num_colors)
    agent_to_color = dict(zip(agent_to_csv_path.keys(), colors))
    for agent, csv_path in agent_to_csv_path.items():
        with open(csv_path) as csvfile:
            reader = csv.DictReader(csvfile)
            data = {'x': [], 'y': []}
            for row in reader:
                if x_axis == 'episode':
                    data['x'].append(int(row['episode']))
                    x_label = 'Episodes'
                    plot_title = 'Average Reward over Episodes'
                elif x_axis == 'training_time':
                    data['x'].append(float(row['training_time']))
                    x_label = 'Training Time'
                    plot_title = 'Average Reward over Time'
                else:
                    raise ValueError('x_axis should be episode or training_time')
                reward = float(row['reward'])
                data['y'].append(reward)

            special_name = name_mapping.get(agent, agent)
            color = agent_to_color[agent]
            sns.lineplot(x='x', y='y', data=data, ax=ax, label=special_name, linewidth=2, color=color)

    ax.set_xlabel(x_label, fontsize=18)
    ax.set_ylabel('Average Reward', fontsize=18)
    ax.set_title(plot_title, fontsize=18)
    ax.legend(fontsize=18, loc='lower right')

    plt.tight_layout()

    save_path = f"experiments/{fig_name}"
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    fig.savefig(save_path, dpi=300)
    plt.show()
    return

def plot_curve(csv_path, save_path, algorithm, x_axis='episode'):
    '''
    Read data from csv file and plot the results
    '''
    with open(csv_path) as csvfile:
        reader = csv.DictReader(csvfile)
        data = {'x': [], 'y': []}
        for row in reader:
            if x_axis == 'episode':
                data['x'].append(int(row['episode']))
                x_label = 'Episodes'
                plot_title = 'Average Reward over Episodes'
            elif x_axis == 'training_time':
                data['x'].append(float(row['training_time']))
                x_label = 'Training Time'
                plot_title = 'Average Reward over Time'
            else:
                raise ValueError('x_axis should be episode or training_time')
            data['y'].append(float(row['reward']))

    sns.set_style("darkgrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    special_name = name_mapping.get(algorithm, algorithm)
    sns.lineplot(x='x', y='y', data=data, ax=ax, label=special_name, linewidth=2)

    ax.set_xlabel(x_label, fontsize=18)
    ax.set_ylabel('Average Reward', fontsize=18)
    ax.set_title(plot_title, fontsize=18)
    ax.legend(fontsize=18)
    ax.legend(fontsize=18, loc='lower right')

    plt.tight_layout()

    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    fig.savefig(save_path, dpi=300)
    return
    # plt.show()

# def plot_curve(csv_path, save_path, algorithm, x_axis='episode'):
#     ''' Read data from csv file and plot the results
#     '''
#     import os
#     import csv
#     import matplotlib.pyplot as plt
#     with open(csv_path) as csvfile:
#         reader = csv.DictReader(csvfile)
#         xs = []
#         ys = []
#         for row in reader:
#             if x_axis == 'episode':
#                 xs.append(int(row['episode']))
#             elif x_axis == 'training_time':
#                 xs.append(int(row['training_time']))
#             else:
#                 raise ValueError('x_axis should be episode or training_time')
#             ys.append(float(row['reward']))
#         fig, ax = plt.subplots()
#         ax.plot(xs, ys, label=algorithm)
#         ax.set(xlabel='episode', ylabel='reward')
#         ax.legend()
#         ax.grid()

#         save_dir = os.path.dirname(save_path)
#         if not os.path.exists(save_dir):
#             os.makedirs(save_dir)
#         fig.show()
#         fig.savefig(save_path)


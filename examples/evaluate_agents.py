''' Modiciation of example code for evaluating trained models in RLCard. This code is designed to evaluate a mixture of agents in the game of Leduc Holdem each against eachother. 
'''
import os
import argparse
import rlcard
from rlcard.utils import (
    set_seed,
    tournament,
    Logger,
    plot_curve,
    plot_curves
)
from rlcard.utils.utils import get_device, get_folders_path, load_config
# importing existing agents
from rlcard.agents.cfr_agent import CFRAgent
from rlcard.agents.dqn_agent import DQNAgent
from rlcard.agents.random_agent import RandomAgent
from rlcard.agents.nfsp_agent import NFSPAgent
# importing our agents
from rlcard.agents.our_agents.NStepSarsaVariantAgent import NStepSarsaVariantAgent
from rlcard.agents.our_agents.QLearningAgent import QLearningAgent
from rlcard.agents.our_agents.Reinforce import ReinforceAgent
# importing our CFR agents
from rlcard.agents.our_agents.CFRAgents.CFRPlusAgent import CFRPlusAgent
from rlcard.agents.our_agents.CFRAgents.CFRVanillaAgent import CFRVanillaAgent
from rlcard.agents.our_agents.CFRAgents.DeepCFRAgent import DeepCFRAgent
from rlcard.agents.our_agents.CFRAgents.MCCFRAgent import MCCFRAgent
# importing our Sarsa Agents
from rlcard.agents.our_agents.SarsaAgents.DeepTrueOnlineSarsaLambda import DeepTrueOnlineSarsaLambdaAgent
from rlcard.agents.our_agents.SarsaAgents.ExpectedSarsaAgent import ExpectedSarsaAgent
from rlcard.agents.our_agents.SarsaAgents.NStepSarsa import NStepSarsaAgent
from rlcard.agents.our_agents.SarsaAgents.SarsaAgent import SarsaAgent
from rlcard.agents.our_agents.SarsaAgents.SarsaLambdaAgent import SarsaLambdaAgent
from rlcard.agents.our_agents.SarsaAgents.SarsaLambdaProbPolicyModTracesWeightsAgent import SarsaLambdaProbPolicyModTracesWeightsAgent
from rlcard.agents.our_agents.SarsaAgents.TrueOnlineSarsaLambda import TrueOnlineSarsaLambdaAgent
# Importing our naive agents
from rlcard.agents.our_agents.NaiveAgents.raise_call_agent import RaiseCallAgent
from rlcard.agents.our_agents.NaiveAgents.check_fold_agent import CheckFoldAgent
from elopy.elo import Elo

import itertools
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List, Dict, Tuple
import numpy as np
from scipy.stats import norm

BASE_FILE_NAME = "experiments_run_10_000"
def calculate_confidence_interval(elo_rating, num_games, confidence_level=0.95, elo_scale=400):
    """
    Calculate the confidence interval for an Elo rating.

    Args:
        elo_rating (float): The Elo rating of the agent.
        num_games (int): The number of games played by the agent.
        confidence_level (float, optional): The desired confidence level. Defaults to 0.95.
        elo_scale (int, optional): The Elo rating scale. Defaults to 400.

    Returns:
        tuple: A tuple containing the lower bound, upper bound, and margin of error of the confidence interval.
    """
    z_score = norm.ppf((1 + confidence_level) / 2)
    margin_of_error = z_score * elo_scale / np.sqrt(num_games)
    lower_bound = elo_rating - margin_of_error
    upper_bound = elo_rating + margin_of_error
    assert lower_bound <= upper_bound
    assert margin_of_error >= 0 and margin_of_error <= 1_000_000
    # confirm no values are nan
    assert not np.isnan(lower_bound)
    assert not np.isnan(upper_bound)
    assert not np.isnan(margin_of_error)
    return lower_bound, upper_bound, margin_of_error

#Likely issue in out update_elo method. I trust an existing library over my implementation, so we do not use this method for evalaution but keep it in code in case we want to later use it.
def update_elo(agent_to_elo: Dict[Tuple, float], player_one, player_two, payoffs_player_one: List[float], k: float = 32, elo_scale: int = 400, initial_elo: float = 1500):
    """
    Update the Elo ratings of the players based on the total observed payoffs for the game.
    
    Args:
        agent_to_elo (Dict[Tuple, float]): Dictionary mapping agent tuples to their current Elo ratings.
        player_one (Agent): Player one in the heads-up match.
        player_two (Agent): Player two in the heads-up match.
        payoffs_player_one (List[float]): List of payoffs for player one in each hand.
        k (float, optional): The k-factor for Elo rating updates. Defaults to 32.
        elo_scale (int, optional): The Elo rating scale. Defaults to 400.
        initial_elo (float, optional): The initial Elo rating for players. Defaults to 1500.
    """
    if (player_one, player_two) not in agent_to_elo:
        agent_to_elo[(player_one, player_two)] = (initial_elo, initial_elo)
    
    player_one_elo, player_two_elo = agent_to_elo[(player_one, player_two)]
    
    # Calculate total payoff for player one across all hands in a game
    total_payoff = sum(payoffs_player_one)

    player_one_elo, player_two_elo = agent_to_elo[(player_one, player_two)]
    
    # Determine the game outcome based on total payoff
    actual_score = 1 if total_payoff > 0 else 0 if total_payoff < 0 else 0.5

    # Calculate expected score for player one using the Elo formula
    expected_score = 1 / (1 + 10 ** ((player_two_elo - player_one_elo) / elo_scale))
    
    # Update Elo ratings based on the overall game outcome
    player_one_elo += k * (actual_score - expected_score)
    player_two_elo -= k * (actual_score - expected_score)
    
    # Update the Elo ratings in the dictionary
    agent_to_elo[(player_one, player_two)] = (player_one_elo, player_two_elo)

    # Ensure the Elo ratings are not negative
    # assert player_one_elo >= 0, f"Player one Elo rating is negative: {player_one_elo}"
    # assert player_two_elo >= 0, f"Player two Elo rating is negative: {player_two_elo}"
    
    # Confirm the Elo ratings are not NaN
    # assert not np.isnan(player_one_elo), f"Player one Elo rating is NaN"
    # assert not np.isnan(player_two_elo), f"Player two Elo rating is NaN"
    return

# This is a mapping of model names from their respective folder names to their desired display names. Not the most elegant code.
name_mapping = {
    "Random" : "Random",
    'CheckFold' : "CheckFold",
    'RaiseCall' : "RaiseCall",
    f"{BASE_FILE_NAME}_leduc_holdem_cfr_result_cfr_model": "CFR Vanilla (Theirs)",
    f"{BASE_FILE_NAME}_leduc_holdem_cfr_vanilla_result_cfr_vanilla_model": "CFR Vanilla",
    f"{BASE_FILE_NAME}_leduc_holdem_mccfrexternal_result_mccfr_model": "MCCFR",
    f"{BASE_FILE_NAME}_leduc_holdem_cfr_plus_result_cfr_plus_model": "CFR+",
    f"{BASE_FILE_NAME}_leduc_holdem_deep_cfraverage_strategy_result_deep_cfr_model": "Deep CFR",
    f"{BASE_FILE_NAME}_leduc_holdem_mccfroutcome_result_mccfr_model": "MCCFR-Outcome",
    f"{BASE_FILE_NAME}_leduc_holdem_mccfrexternal_result_mccfr_model": "MCCFR-External",
    f"{BASE_FILE_NAME}_leduc_holdem_mccfraverage_strategy_result_mccfr_average_strategy": "MCCFR-Avg Strategy",
    f'{BASE_FILE_NAME}_leduc_holdem_deep_cfr_result_deep_cfr_model': 'Deep CFR',
    f'{BASE_FILE_NAME}_leduc_holdem_deep_true_online_sarsa_lambda_result_deep_true_online_sarsa_lambda_model': 'Deep True Online Sarsa Lambda',
    f'{BASE_FILE_NAME}_leduc_holdem_expected_sarsa_result_expected_sarsa_model': 'Expected Sarsa',
    f'{BASE_FILE_NAME}_leduc_holdem_mccfraverage_strategy_result_mccfr_model': 'MCCFR-Avg Strategy',
    f'{BASE_FILE_NAME}_leduc_holdem_mccfrexternal_result_mccfr_model': 'MCCFR-External',
    f'{BASE_FILE_NAME}_leduc_holdem_mccfroutcome_result_mccfr_model': 'MCCFR-Outcome',
    f'{BASE_FILE_NAME}_leduc_holdem_nstep_sarsa_result_nstep_sarsa_model': 'N-Step Sarsa',
    f'{BASE_FILE_NAME}_leduc_holdem_nstep_sarsa_variant_result_nstep_sarsa_variant_model': 'N-Step Sarsa Variant',
    f'{BASE_FILE_NAME}_leduc_holdem_q_learning_result_q_learning_model': 'Q-Learning',
    f'{BASE_FILE_NAME}_leduc_holdem_reinforce_result_reinforce_model': 'REINFORCE',
    f'{BASE_FILE_NAME}_leduc_holdem_sarsa_lambda_prob_policy_mod_traces_weights_result_sarsa_lambda_prob_policy_mod_traces_weights_model': 'Sarsa Lambda Variant',
    f'{BASE_FILE_NAME}_leduc_holdem_sarsa_lambda_result_sarsa_lambda_model': 'Sarsa Lambda',
    f'{BASE_FILE_NAME}_leduc_holdem_sarsa_result_sarsa_model': 'Sarsa',
    f'{BASE_FILE_NAME}_leduc_holdem_true_online_sarsa_lambda_result_true_online_sarsa_lambda_model': 'True Online Sarsa Lambda',
}

def create_bbph_heatmap(results, model_names, name_mapping, save_path='bbph_heatmap.png', sort_by=None):
    """
    Create a heatmap of BBPH values for models against each other.

    Args:
        results (list): The list containing the BBPH values.
        model_names (list): The list of model names.
        name_mapping (dict): The mapping of model names to their desired display names.
        save_path (str): The path to save the heatmap image (default: 'bbph_heatmap.png').
        sort_by (str): The method to sort the columns and rows. Options: 'Avg Reward', 'row_Avg Reward', 'col_Avg Reward', or None (default: None).
    """
    # Convert the results to a DataFrame
    df = pd.DataFrame(results, columns=model_names, index=model_names)

    # Replace model names using the name_mapping
    df.columns = [name_mapping.get(col, col) for col in df.columns]
    df.index = [name_mapping.get(idx, idx) for idx in df.index]

    # Sort the DataFrame based on the specified method
    if sort_by == 'Avg Reward':
        avg_bbph = df.mean(axis=1)
        df_sorted = df.reindex(index=avg_bbph.sort_values(ascending=False).index)
        df_sorted = df_sorted.reindex(columns=avg_bbph.sort_values(ascending=False).index)
    elif sort_by == 'row_Avg Reward':
        row_avg_bbph = df.mean(axis=1)
        df_sorted = df.reindex(index=row_avg_bbph.sort_values(ascending=False).index)
    elif sort_by == 'col_Avg Reward':
        col_avg_bbph = df.mean(axis=0)
        df_sorted = df.reindex(columns=col_avg_bbph.sort_values(ascending=False).index)
    else:
        df_sorted = df

    # Create a larger figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create the heatmap

    heatmap = sns.heatmap(
        df_sorted, 
        annot=True, 
        cmap='coolwarm', 
        cbar_kws={'label': 'BBPH'},  # Removed 'labelsize'
        square=True, 
        fmt='.1f', 
        annot_kws={"fontsize": 12}, 
        linewidths=0.5
    )

    # Adjust the colorbar font size
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=18)

    # Set the axis labels and title
    plt.xlabel('Models', fontsize=18)
    plt.ylabel('Models', fontsize=18)
    if sort_by:
        plt.title(f'Average Big Bind Per Hand\n of Models Against Each Other\n(Sorted by {sort_by})', fontsize=16)
    else:
        plt.title('BBPH of Models Against Each Other', fontsize=16)

    # Rotate the x-axis labels to prevent overlap
    plt.xticks(rotation=45, ha='right', fontsize=14)

    # Adjust the y-axis label font size
    plt.yticks(fontsize=14)

    # Add some padding around the labels
    plt.tight_layout(pad=2.0)

    # Save the heatmap image
    plt.savefig(save_path)

    # Show the plot
    plt.show()

    return fig

# Creates custom agent imports using the best saved checkpoint as defined by each agent's best reward in the training process
def load_model(model_path, env=None, position=None, device=None):
    if os.path.isfile(model_path):  # Torch model
        import torch
        agent = torch.load(model_path, map_location=device)
        agent.set_device(device)
    elif os.path.isdir(model_path):
        model_base_file_name = model_path.split("\\")[-1]
        assert model_base_file_name.endswith("_model")
        model_name = model_base_file_name[:-6] 
        if "cfr_plus" in model_path:
            agent = CFRPlusAgent(
                env,
                model_path,
                load_saved_model=True, save_model=False
            )
        elif "cfr_vanilla" in model_path:
            agent = CFRVanillaAgent(
                env,
                model_path,
                load_saved_model=True, save_model=False
            )
        elif "mccfrexternal" in model_path:
            agent = MCCFRAgent(
                env,
                model_path,
                sampling_strategy='external',
                load_saved_model=True, save_model=False
            )
        elif "mccfroutcome" in model_path:
            agent = MCCFRAgent(
                env,
                model_path,
                sampling_strategy='outcome',
                load_saved_model=True, save_model=False
            )
        elif "mccfraverage" in model_path:
            agent = MCCFRAgent(
                env,
                model_path,
                sampling_strategy='average_strategy',
                load_saved_model=True, save_model=False
            )
        elif "deep_cfr" in model_path:
            agent = DeepCFRAgent(
                env,
                model_path,
                load_saved_model=True, save_model=False
            )
        elif model_path.split("\\")[-1] == "cfr_model":
            agent = CFRAgent(
                env,
                model_path
            ) 
        elif model_name == "sarsa":
            config = load_config(os.path.join('rlcard', 'agents', 'our_agents', 'SarsaAgents', 'configs', 'sarsa_agent_config.json'))
            config['env'] = env
            config['model_path'] = model_path
            agent = SarsaAgent(config)
        elif model_name == "expected_sarsa":
            config = load_config(os.path.join('rlcard', 'agents', 'our_agents', 'SarsaAgents', 'configs', 'expected_sarsa_agent_config.json'))
            config['env'] = env
            config['model_path'] = model_path
            agent = ExpectedSarsaAgent(config)
        elif model_name == "nstep_sarsa":
            config = load_config(os.path.join('rlcard', 'agents', 'our_agents', 'SarsaAgents', 'configs', 'n_step_sarsa_agent_config.json'))
            config['env'] = env
            config['model_path'] = model_path
            agent = NStepSarsaAgent(config)
        elif model_name == "sarsa_lambda":
            config = load_config(os.path.join('rlcard', 'agents', 'our_agents', 'SarsaAgents', 'configs', 'sarsa_lambda_agent_config.json'))
            config['env'] = env
            config['model_path'] = model_path
            agent = SarsaLambdaAgent(config)
        elif model_name == "true_online_sarsa_lambda":
            config = load_config(os.path.join('rlcard', 'agents', 'our_agents', 'SarsaAgents', 'configs', 'true_online_sarsa_lambda_agent_config.json'))
            config['env'] = env
            config['model_path'] = model_path
            agent = TrueOnlineSarsaLambdaAgent(config)
        elif model_name == "deep_true_online_sarsa_lambda":
            config = load_config(os.path.join('rlcard', 'agents', 'our_agents', 'SarsaAgents', 'configs', 'deep_true_online_sarsa_lambda_agent_config.json'))
            config['env'] = env
            config['model_path'] = model_path
            agent = DeepTrueOnlineSarsaLambdaAgent(config)
        elif model_name == "sarsa_lambda_prob_policy_mod_traces_weights":
            config = load_config(os.path.join('rlcard', 'agents', 'our_agents', 'SarsaAgents', 'configs', 'sarsa_lambda_prob_policy_mod_traces_weights_agent_config.json'))
            config['env'] = env
            config['model_path'] = model_path
            agent = SarsaLambdaProbPolicyModTracesWeightsAgent(config)
        elif model_name == "q_learning":
            agent = QLearningAgent(
                env,
                model_path,
                load_saved_model=True, save_model=False
            )
        elif model_name == "nstep_sarsa_variant":
            # model_path = "experiments\\leduc_holdem_nstep_sarsa_variant_result\\nstep_sarsa_variant_model"
            agent = NStepSarsaVariantAgent(
                env,
                model_path = model_path,
                load_saved_model=True, save_model=False
            )
        elif model_name == "reinforce":
            agent = ReinforceAgent(
                env,
                model_path,
                load_saved_model=True, save_model=False
            )
        else:
            raise ValueError(f'Unknown model name {model_name}')
        # load the saved model parameters
        agent.load()
    elif model_path == 'random':
        agent = RandomAgent(num_actions=env.num_actions)
    elif model_path == "check_fold":
        agent = CheckFoldAgent(env.num_actions)
    elif model_path == "raise_call":
        agent = RaiseCallAgent(env.num_actions)
    else:  # A model in the model zoo
        from rlcard import models
        agent = models.load(model_path).agents[position]
    return agent

def evaluate(args):
    # Check whether gpu is available
    device = get_device()
    
    # Seed numpy, torch, random
    set_seed(args.seed)
    
    # Make the environment with seed
    env = rlcard.make(args.env, config={'seed': args.seed})
    
    model_names = []
    for model_path in args.models:
        if "dqn" in model_path:
            model_names.append("DQN")
        # elif "cfr_model" in model_path:
        #     model_names.append("CFR")
        elif model_path == "random":
            model_names.append("Random")
        elif model_path == "check_fold":
            model_names.append("CheckFold")
        elif model_path == "raise_call":
            model_names.append("RaiseCall")
        else:
            # replace // and \\ with _
            model_path = model_path.replace("\\", "_")
            model_path = model_path.replace("/", "_")
            model_names.append(model_path)
    
    model_to_index = {model_name: i for i, model_name in enumerate(model_names)}
    
    num_models = len(args.models)
    payoff_matrix_agents_vs_agents = [[0] * num_models for _ in range(num_models)]
    
    # Load models
    model_combinations = list(itertools.combinations(args.models, 2))
    
    agent_to_elo = {}
    agent_elo_objects = {} 
    agent_elo_results = {}
    ELO_K = 20
    ELO_HCA = 0
    for model_combination in model_combinations:
        # Load models
        agents = []
        agent_names = []
        for position, model_path in enumerate(model_combination):
            model_name_long = model_path.replace("\\", "_")
            model_name = name_mapping.get(model_name_long, model_name_long)
            agent_names.append(model_name)
            loaded_model = load_model(model_path, env, position, device)
            agents.append(loaded_model)
            agent_elo_objects[model_name] = Elo(start_elo=1500, k=ELO_K, hca=ELO_HCA)

            # agent_to_elo[agents] = 1500
        agent_to_elo[tuple(agent_names)] = (1500, 1500)  
        env.set_agents(agents)
        
        print('\nEvaluating model {} vs {}'.format(model_combination[0], model_combination[1]))
        

        rewards, payoffs_player_one = tournament(env, args.num_hands_in_matchup, return_rewards_each_hand = True)
        # split giant tournament into games, where each game is a list of hand payoffs
        assert len(payoffs_player_one) == args.num_hands_in_matchup
        assert len(payoffs_player_one) / args.num_hands_per_game_in_matchup == args.num_desired_games_for_matchup

        # payoffs_player_one = [payoff for payoff in payoffs_player_one if payoff != 0]
        payoffs_player_one_per_game = []
        idx = 0
        while True:
            if idx + args.num_hands_per_game_in_matchup > len(payoffs_player_one):
                break
            payoffs_player_one_per_game.append(payoffs_player_one[idx:idx + args.num_hands_per_game_in_matchup])
            idx += args.num_hands_per_game_in_matchup

        if "check_fold" in agent_names:
            pass
        
        for game_payoffs in payoffs_player_one_per_game:
            update_elo(agent_to_elo, agent_names[-2], agent_names[-1], game_payoffs, k = ELO_K)
            agent_elo_objects[agent_names[-2]].play_game(agent_elo_objects[agent_names[-1]], sum(game_payoffs), is_home=True)

        # Store results in the matrix
        model1_index = model_to_index[model_names[args.models.index(model_combination[0])]]
        model2_index = model_to_index[model_names[args.models.index(model_combination[1])]]
        payoff_matrix_agents_vs_agents[model1_index][model2_index] = rewards[0]
        payoff_matrix_agents_vs_agents[model2_index][model1_index] = rewards[1]

        # Update wins and total games for each agent
        player_id = 0
        wins_player_one = sum([1 for hand_payoffs in payoffs_player_one_per_game if sum(hand_payoffs) > 0])
        wins_player_two =sum([1 for hand_payoffs in payoffs_player_one_per_game if sum(hand_payoffs) < 0])
        # assert wins_player_one + wins_player_two == args.num_desired_games_for_matchup
        for agent_name, reward in zip(agent_names, rewards):
            if agent_name not in agent_elo_results:
                agent_elo_results[agent_name] = {
                    "Elo Our Implementation": agent_to_elo[tuple(agent_names)][agent_names.index(agent_name)],
                    "Elo Library": agent_elo_objects[agent_name].elo,
                    "Wins": 0,
                    "Total Games": 0
                }
            if player_id ==0:
                agent_elo_results[agent_name]["Wins"] += wins_player_one
            else:
                agent_elo_results[agent_name]["Wins"] += wins_player_two
            agent_elo_results[agent_name]["Total Games"] += wins_player_one + wins_player_two # args.num_desired_games_for_matchup
            player_id += 1

    # Calculate confidence intervals and win percentages
    for agent_name, results in agent_elo_results.items():
        num_games = results["Total Games"]
        lower_bound, upper_bound, margin_of_error = calculate_confidence_interval(results["Elo Our Implementation"], num_games)
        agent_elo_results[agent_name]["CI"] = (round(lower_bound, 2), round(upper_bound, 2))
        agent_elo_results[agent_name]["MoE"] = margin_of_error
        win_percentage = results["Wins"] / num_games * 100
        agent_elo_results[agent_name]["Win Percentage"] = win_percentage
        
        # Assert statements to catch bugs
        assert not np.isnan(lower_bound), f"Lower bound is NaN for agent: {agent_name}"
        assert not np.isnan(upper_bound), f"Upper bound is NaN for agent: {agent_name}"
        assert 0 <= win_percentage <= 100, f"Win percentage is not between 0 and 100 for agent: {agent_name}"

    # Sort agents by elo and print their elo sorted
    sorted_agent_elo_results = {k: v for k, v in sorted(agent_elo_results.items(), key=lambda item: item[1]['Elo Library'], reverse=True)}
    for agent_name, results in sorted_agent_elo_results.items():
        print(f"Agent: {agent_name}")
        print(f"  Elo Our Implementation: {results['Elo Our Implementation']}")
        print(f"  Elo Library: {results['Elo Library']}")
        print(f"  Confidence Interval: {results['CI']}")
        print(f"  Wins: {results['Wins']}")
        print(f"  Win Percentage: {results['Win Percentage']:.2f}%\n")
        print()

    # Save results to a file
    with open("agent_elog_results.txt", "w") as file:
        for agent_name, results in sorted_agent_elo_results.items():
            file.write(f"Agent: {agent_name}\n")
            file.write(f"  Elo Our Implementation: {results['Elo Our Implementation']}\n")
            file.write(f"  Elo Library: {results['Elo Library']}\n")
            file.write(f"  Margin of Error: {results['MoE']}\n")
            file.write(f"  Confidence Interval: {results['CI']}\n")
            file.write(f"  Wins: {results['Wins']}\n")
            file.write(f"  Win Percentage: {results['Win Percentage']:.2f}%\n")
            file.write("\n")

    create_bbph_heatmap(payoff_matrix_agents_vs_agents, model_names, name_mapping, save_path=f'{BASE_FILE_NAME}_/overall_bbph_heatmap.png', sort_by='Avg Reward')
    return payoff_matrix_agents_vs_agents, agent_elo_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Evaluation example in RLCard")
    parser.add_argument(
        '--env',
        type=str,
        default='leduc-holdem',
        choices=[
            'blackjack',
            'leduc-holdem',
            'limit-holdem',
            'doudizhu',
            'mahjong',
            'no-limit-holdem',
            'uno',
            'gin-rummy',
        ],
    )
    parser.add_argument(
        '--models',
        nargs='*',
        default=[
            f'{BASE_FILE_NAME}_leduc_holdem_dqn_result/model.pth',
            'random',
        ],
    )
    parser.add_argument(
        '--cuda',
        type=str,
        default='',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
    )
    parser.add_argument(
        '--num_hands_in_matchup',
        type=int,
        default=100_000,
    )
    parser.add_argument(
        '--num_desired_games_for_matchup',
        type=int,
        default=1_000,
    )
    parser.add_argument(
        '--num_hands_per_game_in_matchup',
        type=int,
        default=100,
    )
    args = parser.parse_args()
    args.num_hands_in_matchup = 2_000
    args.num_hands_per_game_in_matchup = 100
    args.num_desired_games_for_matchup = 20

    args.num_desired_games_for_matchup = args.num_hands_in_matchup // args.num_hands_per_game_in_matchup
    print(f"Number of hands per game: {args.num_desired_games_for_matchup}")
    assert args.num_hands_in_matchup // args.num_hands_per_game_in_matchup == args.num_desired_games_for_matchup
    args.env = "leduc-holdem"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    args.models = get_folders_path(base_folder =BASE_FILE_NAME, num_levels_desired = 2)
    args.models.extend(['random', 'check_fold', 'raise_call'])

    # Removing these models due to poor performance, to speed up evaluation, and do reduce clutter on 2x2 matrix visualization
#     poor_performing_models = ['experiments\\leduc_holdem_mccfraverage_strategy_result\\mccfr_model','check_fold',
# 'experiments\\leduc_holdem_nstep_sarsa_variant_result\\nstep_sarsa_variant_model', 'experiments\\leduc_holdem_sarsa_lambda_prob_policy_mod_traces_weights_result\\sarsa_lambda_prob_policy_mod_traces_weights_model'
# ]
    poor_performing_models = []
    # Removing the following due to poor performance
    for model_to_remove in poor_performing_models:
        if model_to_remove in args.models:
            args.models.remove(model_to_remove)
        else:
            raise ValueError(f"Model {model_to_remove} not found in args.models")
    start_evaluation_time = time.time()
    evaluate(args)
    end_evaluation_time = time.time()
    print(f"Total evaluation time: {(end_evaluation_time - start_evaluation_time)//60} minutes.")


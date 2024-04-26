'''
This file, train_agents.py, is used to train all agents on the game of Leduc Holdem. It is built upon existing example code from the RLCard library to train only their CFR agent. We extend this code to train a variety of different agents including CFR and Sarsa Variants
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
from rlcard.utils.utils import load_config, get_evaluation_iterations, plot_curves_line_types, plot_curves_numbered_lines, plot_curves_variant_grouped
from tqdm import tqdm
import time
import random
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
import json
import matplotlib.ticker as ticker

def train(args, agent, env, eval_env):
    """
    Train the agent using the specified environment.

    Args:
        args (object): The arguments object containing training configurations.
        agent (object): The agent object to be trained.
        env (object): The training environment.
        eval_env (object): The evaluation environment.

    Returns:
        str: The path to the CSV file containing the training performance data.
        str: The path to the figure file showing the learning curve.
    """
    
    # Future Optimization: Show training reward against pretrained agent, not just RandomAgent. We don't get a strong enough signal in later epochs by comparing only to RandomAgent in training.
    eval_env.set_agents([
        agent,
        RandomAgent(num_actions=env.num_actions),
    ])

    training_time_minutes = 0
    # Start training
    with Logger(args.log_dir) as logger:
        episode = 0
        while (args.train_by_time and training_time_minutes < args.training_time_minutes) or (not args.train_by_time and episode < args.num_episodes):  
            training_start_time_for_episode = time.time()
            agent.train()
            training_end_time_for_episode = time.time()
            training_time_minutes += (training_end_time_for_episode - training_start_time_for_episode)/60
            # Evaluate the performance. Play with Random agents.
            if episode in args.evaluation_iterations:
            # if episode % args.evaluate_every == 0:
                agent.save() # Save model
                # We add a random value to the number of games to play to add reduce likelyhood of ties in the payoffs between two agents to prevent overlapping lines in the plotting of the training reward curves. This is not used in evaluation of the agent's performance. See evaluate_agents.py for evaluation of the agent's performance.
                random_val = random.randint(-args.num_eval_games // 10, args.num_eval_games // 10)
                reward = payoffs = bb_per_hand = tournament(
                        eval_env,
                        args.num_eval_games + random_val
                    )[0]

                if hasattr(agent, 'save_only_best'):
                    agent.save_only_best(reward)

                logger.log_performance(
                    episode,
                    payoffs,
                    training_time=training_time_minutes
                )
            episode += 1
        # Get the paths
        csv_path, fig_path = logger.csv_path, logger.fig_path
    # Plot the learning curve
    # plot_curve(csv_path, fig_path, 'cfr_vanilla', x_axis='episode')
    # plot_curve(csv_path, fig_path, 'cfr_vanilla', x_axis='training_time')
    return csv_path, fig_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser("CFR example in RLCard")
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
    )
    parser.add_argument(
        '--num_episodes',
        type=int,
        default=10_000, 
    )
    parser.add_argument(
        '--num_eval_games',
        type=int,
        default=1_000, 
    )
    parser.add_argument(
        '--evaluation_iterations',
        type=list,
        default=[], 
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='experiments/leduc_holdem_cfr_result/',
    )
    parser.add_argument(
        '--agents_to_train',
        type=str,
        default=['cfr', 'cfr_vanilla', 'cfr_plus','mccfr_outcome','mccfr_external','mccfr_average', 'deep_cfr', "sarsa", 'expected_sarsa', "nstep_sarsa", "nstep_sarsa_variant", "sarsa_lambda","sarsa_lambda_prob_policy_mod_traces_weights", "true_online_sarsa_lambda", 'deep_true_online_sarsa_lambda',  "q_learning", 'reinforce'],
    )
    parser.add_argument( 
        '--perform_debug_logging',
        type=bool,
        default=False,
    )
    parser.add_argument(
        '--sampling_strategy',
        type=str,
        default='',
    )
    parser.add_argument(
        '--train_by_time',
        type=bool,
        default=True,
    )
    parser.add_argument(
        '--training_time_minutes',
        type=int,
        default=60,  # Training time in minutes
    )
    args = parser.parse_args()
    # Seed numpy, torch, random
    set_seed(args.seed)

    args.train_by_time = True
    args.training_time_minutes = 60
    # args.num_episodes = 10# TODO Ultamitely Delete this
    # args.num_eval_games = 10 # Ultamitely Delete this
    args.evaluation_iterations = get_evaluation_iterations(args.num_episodes)

    print(f"Training {args.agents_to_train} on Leduc Holdem")
    training_start_time_all_agents = time.time()
    agent_to_csv_path, agent_to_fig_path = {}, {}
    for idx, agent_to_train in enumerate(args.agents_to_train):
        if 'mccfr' in agent_to_train:
            if "outcome" in agent_to_train:
                 args.sampling_strategy = "outcome"
            elif "external" in agent_to_train:
                args.sampling_strategy = "external"
            elif "average" in agent_to_train:
                args.sampling_strategy = "average_strategy"
            else:
                raise ValueError(f"Not recoginizing sampling strategy for MCCFR Agent")
            agent_to_train = "mccfr"
            args.log_dir = f'experiments/leduc_holdem_{agent_to_train}{args.sampling_strategy}_result/'
        else:
            args.log_dir = f'experiments/leduc_holdem_{agent_to_train}_result/'
        # Note: this code leaves much to be deisred, but we keep the condition style as is to match RLCard's existing code. This should be optimized further.
        # Make environments, CFR only supports Leduc Holdem
        env = rlcard.make(
            'leduc-holdem',
            config={
                'seed': 0,
                'allow_step_back': True,
            }
        )

        eval_env = rlcard.make(
            'leduc-holdem',
            config={
                'seed': 0,
            }
        )

        if agent_to_train == 'cfr':
            agent = CFRAgent(
                env,
                os.path.join(
                    args.log_dir,
                    'cfr_model',
                ),
            )

        elif agent_to_train == 'cfr_vanilla':
            agent = CFRVanillaAgent(
                env,
                os.path.join(
                    args.log_dir,
                    'cfr_vanilla_model',
                ),
                perform_logging=args.perform_debug_logging,
                load_saved_model=True, save_model=True

            )

        elif agent_to_train == 'cfr_plus':
            agent = CFRPlusAgent(
                env,
                os.path.join(
                    args.log_dir,
                    'cfr_plus_model',
                ),
                perform_logging=args.perform_debug_logging,
                load_saved_model=False, save_model=True

            )

        elif agent_to_train == 'mccfr':
            agent = MCCFRAgent(
                env,
                os.path.join(
                    args.log_dir,
                    'mccfr_model',
                ),
                perform_logging=args.perform_debug_logging,
                sampling_strategy=args.sampling_strategy,
                load_saved_model=True, save_model=True

            )

        elif agent_to_train == 'deep_cfr':
            agent = DeepCFRAgent(
                env,
                os.path.join(
                    args.log_dir,
                    'deep_cfr_model',
                ),
                perform_logging=args.perform_debug_logging,
                load_saved_model=True, save_model=True
            )

        elif agent_to_train == 'q_learning':
            agent = QLearningAgent(
                env,
                os.path.join(
                    args.log_dir,
                    'q_learning_model',
                ),
                perform_logging=args.perform_debug_logging,
                load_saved_model=True, save_model=True
            )

        elif agent_to_train == 'sarsa':
            config = load_config(os.path.join('rlcard', 'agents', 'our_agents', 'SarsaAgents', 'configs', 'sarsa_agent_config.json'))
            config['env'] = env
            config['model_path'] = os.path.join(args.log_dir, 'sarsa_model')
            agent = SarsaAgent(config)

        elif agent_to_train == 'expected_sarsa':
            config = load_config(os.path.join('rlcard', 'agents', 'our_agents', 'SarsaAgents', 'configs', 'expected_sarsa_agent_config.json'))
            config['env'] = env
            config['model_path'] = os.path.join(args.log_dir, 'expected_sarsa_model')
            agent = ExpectedSarsaAgent(config)

        elif agent_to_train == 'nstep_sarsa':
            config = load_config(os.path.join('rlcard', 'agents', 'our_agents', 'SarsaAgents', 'configs', 'n_step_sarsa_agent_config.json'))
            config['env'] = env
            config['model_path'] = os.path.join(args.log_dir, 'nstep_sarsa_model')
            agent = NStepSarsaAgent(config)

        elif agent_to_train == 'nstep_sarsa_variant':
            agent = NStepSarsaVariantAgent(
                env,
                os.path.join(
                    args.log_dir,
                    'nstep_sarsa_variant_model',
                ),
                perform_logging=args.perform_debug_logging,
                load_saved_model=True, save_model=True
            )

        elif agent_to_train == 'sarsa_lambda':
            config = load_config(os.path.join('rlcard', 'agents', 'our_agents', 'SarsaAgents', 'configs', 'sarsa_lambda_agent_config.json'))
            config['env'] = env
            config['model_path'] = os.path.join(args.log_dir, 'sarsa_lambda_model')
            agent = SarsaLambdaAgent(config)

        elif agent_to_train == 'true_online_sarsa_lambda':
            config = load_config(os.path.join('rlcard', 'agents', 'our_agents', 'SarsaAgents', 'configs', 'true_online_sarsa_lambda_agent_config.json'))
            config['env'] = env
            config['model_path'] = os.path.join(args.log_dir, 'true_online_sarsa_lambda_model')
            agent = TrueOnlineSarsaLambdaAgent(config)

        elif agent_to_train == 'deep_true_online_sarsa_lambda':
            config = load_config(os.path.join('rlcard', 'agents', 'our_agents', 'SarsaAgents', 'configs', 'deep_true_online_sarsa_lambda_agent_config.json'))
            config['env'] = env
            config['model_path'] = os.path.join(args.log_dir, 'deep_true_online_sarsa_lambda_model')
            agent = DeepTrueOnlineSarsaLambdaAgent(config)

        elif agent_to_train == 'sarsa_lambda_prob_policy_mod_traces_weights':
            config = load_config(os.path.join('rlcard', 'agents', 'our_agents', 'SarsaAgents', 'configs', 'sarsa_lambda_prob_policy_mod_traces_weights_agent_config.json'))
            config['env'] = env
            config['model_path'] = os.path.join(args.log_dir, 'sarsa_lambda_prob_policy_mod_traces_weights_model')
            agent = SarsaLambdaProbPolicyModTracesWeightsAgent(config)

        elif agent_to_train == 'reinforce':
            agent = ReinforceAgent(
                env,
                os.path.join(
                    args.log_dir,
                    'reinforce_model',
                ),
                perform_logging=args.perform_debug_logging,
                load_saved_model=True, save_model=True
            )
        else:
            raise ValueError(f"Agent name to train was not one of the recognized CFR agents to train")
        csv_path, fig_path = train(args, agent, env, eval_env)
        agent_to_csv_path[args.agents_to_train[idx]] = csv_path
        agent_to_fig_path[args.agents_to_train[idx]] = fig_path

    training_end_time_all_agents = time.time()
    print(f"Training all agents took {training_end_time_all_agents - training_start_time_all_agents} seconds")

    # # Plot All Agents
    # plot_curves_line_types(agent_to_csv_path, x_axis='episode', fig_name="all_agents.png")
    # plot_curves_line_types(agent_to_csv_path, x_axis='training_time', fig_name="all_agents.png")

    # plot_curves_numbered_lines(agent_to_csv_path, x_axis='episode', fig_name="all_agents.png")
    # plot_curves_numbered_lines(agent_to_csv_path, x_axis='training_time', fig_name="all_agents.png")

    # plot_curves_variant_grouped(agent_to_csv_path, x_axis='episode', fig_name="all_agents.png")
    # plot_curves_variant_grouped(agent_to_csv_path, x_axis='training_time', fig_name="all_agents.png")


    # plot_curves(agent_to_csv_path, x_axis='episode', fig_name = "all_agents_episode.png")
    # plot_curves(agent_to_csv_path, x_axis='training_time', fig_name = "all_agents_training_time.png")

    # # CFR Agents
    # cfr_agents = ['cfr', 'cfr_vanilla', 'cfr_plus', 'mccfr_outcome', 'mccfr_external', 'mccfr_average', 'deep_cfr']
    # agent_to_csv_path_cfr = {k: v for k, v in agent_to_csv_path.items() if k in cfr_agents}
    # plot_curves(agent_to_csv_path_cfr, x_axis='episode', fig_name = "cfr_agents_episode.png")
    # plot_curves(agent_to_csv_path_cfr, x_axis='training_time', fig_name = "cfr_agents_training_time.png")

    # # Sarsa Agents
    # sarsa_agents = ['sarsa', 'expected_sarsa', 'nstep_sarsa', "nstep_sarsa_variant", 'sarsa_lambda', 'true_online_sarsa_lambda', 'deep_true_online_sarsa_lambda', 'sarsa_lambda_prob_policy_mod_traces_weights']
    # agent_to_csv_path_sarsa = {k: v for k, v in agent_to_csv_path.items() if k in sarsa_agents}
    # plot_curves(agent_to_csv_path_sarsa, x_axis='episode', fig_name = "sarsa_agents_episode.png")

    # # Other Agents
    # other_agents = ['q_learning', 'reinforce']
    # agent_to_csv_path_other = {k: v for k, v in agent_to_csv_path.items() if k in other_agents}
    # plot_curves(agent_to_csv_path_other, x_axis='episode', fig_name = "other_agents_episode.png")

    # Save agent_to_csv_path as a json file for easy import if in plot only mode!
    with open('agent_to_csv_path.json', 'w') as f:
        json.dump(agent_to_csv_path, f)


    print(f"Training Complete. Exiting...")

        

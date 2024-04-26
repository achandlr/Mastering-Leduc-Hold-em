import json
from rlcard.utils.utils import plot_curves, plot_curves_line_types, plot_curves_numbered_lines, plot_curves_variant_grouped
# Load agent_to_csv_path from json file\
with open('agent_to_csv_path.json', 'r') as f:
    agent_to_csv_path = json.load(f)

# Plot All Agents
plot_curves_line_types(agent_to_csv_path, x_axis='episode', fig_name="all_agents.png")
plot_curves_line_types(agent_to_csv_path, x_axis='training_time', fig_name="all_agents.png")

plot_curves_numbered_lines(agent_to_csv_path, x_axis='episode', fig_name="all_agents.png")
plot_curves_numbered_lines(agent_to_csv_path, x_axis='training_time', fig_name="all_agents.png")

plot_curves_variant_grouped(agent_to_csv_path, x_axis='episode', fig_name="all_agents.png")
plot_curves_variant_grouped(agent_to_csv_path, x_axis='training_time', fig_name="all_agents.png")


plot_curves(agent_to_csv_path, x_axis='episode', fig_name = "all_agents_episode.png")
plot_curves(agent_to_csv_path, x_axis='training_time', fig_name = "all_agents_training_time.png")

# CFR Agents
cfr_agents = ['cfr', 'cfr_vanilla', 'cfr_plus', 'mccfr_outcome', 'mccfr_external', 'mccfr_average', 'deep_cfr']
agent_to_csv_path_cfr = {k: v for k, v in agent_to_csv_path.items() if k in cfr_agents}
plot_curves(agent_to_csv_path_cfr, x_axis='episode', fig_name = "cfr_agents_episode.png")
plot_curves(agent_to_csv_path_cfr, x_axis='training_time', fig_name = "cfr_agents_training_time.png")

# Sarsa Agents
sarsa_agents = ['sarsa', 'expected_sarsa', 'nstep_sarsa', "nstep_sarsa_variant", 'sarsa_lambda', 'true_online_sarsa_lambda', 'deep_true_online_sarsa_lambda', 'sarsa_lambda_prob_policy_mod_traces_weights']
agent_to_csv_path_sarsa = {k: v for k, v in agent_to_csv_path.items() if k in sarsa_agents}
plot_curves(agent_to_csv_path_sarsa, x_axis='episode', fig_name = "sarsa_agents_episode.png")

# Other Agents
other_agents = ['q_learning', 'reinforce']
agent_to_csv_path_other = {k: v for k, v in agent_to_csv_path.items() if k in other_agents}
plot_curves(agent_to_csv_path_other, x_axis='episode', fig_name = "other_agents_episode.png")

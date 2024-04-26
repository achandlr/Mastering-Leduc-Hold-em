# import rlcard
# from rlcard.agents import RandomAgent

# env = rlcard.make('blackjack')
# env.set_agents([RandomAgent(num_actions=env.num_actions)])

# print(env.num_actions) # 2
# print(env.num_players) # 1
# print(env.state_shape) # [[2]]
# print(env.action_shape) # [None]

# trajectories, payoffs = env.run()

import rlcard
from rlcard import models
from rlcard.agents import LeducholdemHumanAgent as HumanAgent
from rlcard.utils import print_card

# Make environment
env = rlcard.make('leduc-holdem')
human_agent = HumanAgent(env.num_actions)
cfr_agent = models.load('leduc-holdem-cfr').agents[0]
env.set_agents([
    human_agent,
    cfr_agent,
])

print(">> Leduc Hold'em pre-trained model")

while (True):
    print(">> Start a new game")

    trajectories, payoffs = env.run(is_training=False)
    # If the human does not take the final action, we need to
    # print other players action
    final_state = trajectories[0][-1]
    action_record = final_state['action_record']
    state = final_state['raw_obs']
    _action_list = []
    for i in range(1, len(action_record)+1):
        # break once we reach the action of the current player
        if action_record[-i][0] == state['current_player']:
            break
        _action_list.insert(0, action_record[-i])
    for pair in _action_list:
        print('>> Player', pair[0], 'chooses', pair[1])

    # Let's take a look at what the agent card is
    print('===============     CFR Agent    ===============')
    print_card(env.get_perfect_information()['hand_cards'][1])
    # TODO: Below is a misrepresentation of the payoff. The payoff does not represent chips won but rather net_gained chips / big blind. Maybe, issue can be fixed by *=2
    print('===============     Result     ===============')
    if payoffs[0] > 0:
        print(f'You win {payoffs[0]} big blinds, and {payoffs[0]*2} chips')
    elif payoffs[0] == 0:
        print('It is a tie.')
    else:
        print(f'You lose {-payoffs[0]} big blinds, and {-payoffs[0]*2} chips')
    print('')

    input("Press any key to continue...")
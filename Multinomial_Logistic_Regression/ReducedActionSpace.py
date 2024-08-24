import retro
import random
import numpy as np
def practise():

    env = retro.make(game="MortalKombatII-Genesis", use_restricted_actions = retro.Actions.FILTERED)
    env.reset()
    done = False
    num_actions = env.action_space.n  # num_action its actually 12 Its MultiBinary
    all_actions = [np.binary_repr(i, width=num_actions) for i in range(
        2 ** num_actions)]  # np.binary_reprconvert an integer into its binary representation as a string and width is output binary length,this line takes all action from 000000000000 to 111111111111

    actions = []  # for all actions of player
    bin_action = []  # for all binary actions of player
    total_act_num = 0  # number  of actions of player

    for index, binary_action in enumerate(
            all_actions):  # enumerate keeping track of both the elements and their corresponding indices  000000000001  000000000010
        if (env.unwrapped.get_action_meaning(binary_action) != [] and env.unwrapped.get_action_meaning(binary_action) not in actions
                and not (env.unwrapped.get_action_meaning(binary_action).__contains__('Z') or env.unwrapped.get_action_meaning(
                binary_action).__contains__('Y') or env.unwrapped.get_action_meaning(binary_action).__contains__(
                'X'))):  # if there is no meaning of an binary action then reject them
            print(f"Action ", index, binary_action, env.unwrapped.get_action_meaning(binary_action))
            actions.append(env.unwrapped.get_action_meaning(
                binary_action))  # append the meaning of action in actions list to check if the same action come again then if statement reject them
            bin_action.append(binary_action)
            total_act_num = total_act_num + 1

    print("Total actions is ", total_act_num)

    for i in range(1):
        while not done:
            if done:
                obs = env.reset()
            env.render()
            act=random.choice(bin_action)
            print(act)
            print(env.unwrapped.get_action_meaning(act))
            obs, reward, done, info = env.step(act)

practise()


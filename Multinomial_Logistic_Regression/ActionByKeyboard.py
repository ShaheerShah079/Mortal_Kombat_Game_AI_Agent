import retro
import time
import keyboard
import numpy as np
import random

def main():
    env = retro.make(game="MortalKombatII-Genesis", use_restricted_actions = retro.Actions.FILTERED)
    env.reset()
    done = False


    num_actions = env.action_space.n  #num_action its actually 12 Its MultiBinary
    all_actions = [np.binary_repr(i, width=num_actions) for i in range(2 ** num_actions)]  #np.binary_reprconvert an integer into its binary representation as a string and width is output binary length,this line takes all action from 000000000000 to 111111111111

    actions=[]       # for all actions of player
    bin_action=[]    # for all binary actions of player
    total_act_num=0  # number  of actions of player

    for index, binary_action in enumerate(all_actions): #enumerate keeping track of both the elements and their corresponding indices
        if env.unwrapped.get_action_meaning(binary_action) != [] and env.unwrapped.get_action_meaning(binary_action) not in actions:  # if there is no meaning of an binary action then reject them
            print(f"Action ", index, binary_action, env.unwrapped.get_action_meaning(binary_action))
            actions.append(env.unwrapped.get_action_meaning(binary_action))  #append the meaning of action in actions list to check if the same action come again then if statement reject them
            bin_action.append(binary_action)
            total_act_num=total_act_num+1

    print("Total actions is ",total_act_num)
    for i in range(1):
        while not done:
            if done:
                obs = env.reset()
            env.render()
            if keyboard.is_pressed('w'):
                obs, reward, done, info = env.step(np.array([1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1]))
                print(env.unwrapped.get_action_meaning(np.array([1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1])))
            elif keyboard.is_pressed('a'):
                obs, reward, done, info = env.step(np.array([1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1]))
                print(env.unwrapped.get_action_meaning(np.array([1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1])))
            elif keyboard.is_pressed('s'):
                obs, reward, done, info = env.step(np.array([1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0]))
                print(env.unwrapped.get_action_meaning(np.array([1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0])))
            elif keyboard.is_pressed('d'):
                obs, reward, done, info = env.step(np.array([1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1]))
                print(env.unwrapped.get_action_meaning(np.array([1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1])))
            elif keyboard.is_pressed('j'):
                obs, reward, done, info = env.step(np.array([1,0,1,1,1,1,1,1,0,0,0,0]))
                print(env.unwrapped.get_action_meaning(np.array([1,0,1,1,1,1,1,1,0,0,0,0])))
            elif keyboard.is_pressed('k'):
                obs, reward, done, info = env.step(np.array([0,0,1,1,1,1,1,1,1,0,0,0]))
                print(env.unwrapped.get_action_meaning(np.array([0,0,1,1,1,1,1,1,1,0,0,0])))
            elif keyboard.is_pressed('i'):
                obs, reward, done, info = env.step(np.array([0,1,1,1,1,1,1,1,0,0,0,0]))
                print(env.unwrapped.get_action_meaning(np.array([0,1,1,1,1,1,1,1,0,0,0,0])))
            else:
                obs, reward, done, info = env.step(np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1]))
                print(env.unwrapped.get_action_meaning(np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1])))
            time.sleep(0.005)


            # no operation:
            #np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1])

            #a,down.x
            #[0 ,1 ,0 ,1, 0, 1, 0, 0, 0, 0, 1, 0]


            #b,c
            #[1 ,0, 1, 1, 0, 0, 0 ,0 ,1 ,0 ,0 ,0]))

            #down,right,y
            # obs, reward, done, info = env.step(np.array([0 ,0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0]))

            #B 101111110000 kick

            #left 101111100011

            # a 011111110000 punch

            #c 001111111000 kick

            #z 001011000001 nothing

            #x 001011000010 nothing

            #y 001011000100 nothing

            #left 101111100011

            #down 000101111101

main()


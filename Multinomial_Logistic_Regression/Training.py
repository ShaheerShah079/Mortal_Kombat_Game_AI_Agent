import retro
import numpy as np
from PIL import Image
import random
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def practiseImages():
    env = retro.make(game="MortalKombatII-Genesis", use_restricted_actions = retro.Actions.FILTERED)
    env.reset()
    d=False

    num_actions = env.action_space.n
    all_actions = [np.binary_repr(i, width=num_actions) for i in range(
        2 ** num_actions)]  # np.binary_reprconvert an integer into its binary representation as a string and width is output binary length,this line takes all action from 000000000000 to 111111111111

    actions = []  # for all actions of player
    bin_action = []  # for all binary actions of player
    total_act_num = 0  # number  of actions of player

    for index, binary_action in enumerate(
            all_actions):  # enumerate keeping track of both the elements and their corresponding indices  000000000001  000000000010
        if (env.unwrapped.get_action_meaning(binary_action) != [] and env.unwrapped.get_action_meaning(
                binary_action) not in actions
                and not (env.unwrapped.get_action_meaning(binary_action).__contains__(
                    'Z') or env.unwrapped.get_action_meaning(
                    binary_action).__contains__('Y') or env.unwrapped.get_action_meaning(binary_action).__contains__(
                    'X'))):  # if there is no meaning of an binary action then reject them
            print(f"Action ", index, binary_action, env.unwrapped.get_action_meaning(binary_action))
            actions.append(env.unwrapped.get_action_meaning(
                binary_action))  # append the meaning of action in actions list to check if the same action come again then if statement reject them
            bin_action.append(binary_action)
            total_act_num = total_act_num + 1

    print("Total actions is ", total_act_num)

    image_stack=[]
    action_stack=[]
    dataimg=[]
    dataaction=[]
    steps=0
    obs = env.reset()
    while steps <= 1000:
        env.render()
        act = random.choice(bin_action)
        obs, reward, done, info = env.step(act)
        if info['health']==0 or info['enemy_health']==0:
            env.reset()
        image_stack.append(obs)
        action_stack.append(act)
        if reward > 5:
            if (len(image_stack) > 16):
                for i in range(-15, -2, 1):
                    dataimg.append(np.array(Image.fromarray(image_stack[i]).convert('L')).flatten())
                    dataaction.append(int(str(''.join(map(str, action_stack[i]))), 2))
            else:
                for i in range(len(image_stack) - 3):
                    dataimg.append(np.array(Image.fromarray(image_stack[i]).convert('L')).flatten())
                    dataaction.append(int(str(''.join(map(str, act))), 2))
            image_stack = []

            steps += 1
            print(steps)
    dataimg=np.array(dataimg)
    x1,x2,y1,y2=train_test_split(dataimg,dataaction,test_size=0.2)
    # train_test_split(dataimg,)
    # print(np.array(dataimg).ndim)
    # print(type(np.array(dataimg)))
    #
    # print(np.array(dataaction).ndim)
    # print(type(np.array(dataaction)))
    # plt.imshow(pandaData['image'][0])
    # plt.show()
    # x_train, x_test, y_train, y_test = train_test_split(dataimg,dataaction, test_size=0.2)
    reg = LogisticRegression(max_iter=1000, solver='sag')
    scaler = StandardScaler()
    x1= scaler.fit_transform(x1)
    reg.fit(x1,y1)
    print("acuracy ",reg.score(x2,y2))
    print(reg.intercept_)
    print("Converged:", reg.converged_)
    joblib.dump(reg,'modelWithTenThousandImages3')

practiseImages()


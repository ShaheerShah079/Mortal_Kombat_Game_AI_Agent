import random
import retro
import pandas as pd
from PIL import Image
import time
import actions as game_actions

env = retro.make('MortalKombatII-Genesis')
actions=[]
def render():
    obs = env.reset()
    observations=[]
    images_path=[]
    actions=[]
    action_meaning=[]
    decimal_actions=[]
    temp_obs=[]
    temp_actions=[]
    for i in range(10000):
        env.render()
        if i%20==0:
         action=random.choice(game_actions.actions)
         # print(env.unwrapped.get_action_meaning(action))
         # while env.unwrapped.get_action_meaning(action)==[]:
         #    action = env.action_space.sample()
        obs, reward, done, info= env.step(action)
        if reward>0:
            print(reward)
        if info['health']==0 :
            env.reset()
        temp_actions.append(action)
        temp_obs.append(obs)

        if reward>5:
            print(reward)
            for k in range(15, 3, -1):
                observations.append(temp_obs[i-k])
                # string = ''
                # for item in temp_actions[i-k]:
                #     if item != '[' and item != ']':
                #        string = string + str(item)
                # actions.append(string)
                decimal_actions.append(int(string, 2))
                action_meaning.append(env.unwrapped.get_action_meaning(temp_actions[i-k]))
                

    counter=12944
    for i in range(len(observations)):
        path='MK2/N/DatasetImages/img_'+str(counter)+'.png';
        img=Image.fromarray(observations[i])
        img.save(path)
        images_path.append(path)
        counter+=1

    new_data=pd.DataFrame({
        'Images':images_path,
        'decimal Actions':decimal_actions,
        'Alpha Action':action_meaning
    })

    old_data=pd.read_csv('MK2/N/Dataset.csv')
    new_data=pd.concat([old_data,new_data],ignore_index=True)
    new_data.to_csv('MK2/N/Dataset.csv')

render()
import retro
import joblib
import numpy as np
from PIL import Image
import time
def practise():
    global action
    reg = joblib.load("modelWithTenThousandImages2")
    env = retro.make(game="MortalKombatII-Genesis", use_restricted_actions = retro.Actions.FILTERED)
    print(reg.intercept_)

    # obs=env.reset()
    # img=np.array(Image.fromarray(obs).convert('L')).flatten()
    # print(img.ndim)
    # img=img.reshape(1,len(img))
    # print(img.ndim)
    # action=np.array(list(np.binary_repr(int(reg.predict(img)))), dtype=int)
    # print(type(action))
    win=0
    j=0
    for i in range(10):
        done = False
        obs = env.reset()
        while not done:
            env.render()
            if j%10==0:
                img = np.array(Image.fromarray(obs).convert('L')).flatten()
                img = img.reshape(1, len(img))
                action = np.array(list(np.binary_repr(int(reg.predict(img)))), dtype=int)
            # print(env.unwrapped.get_action_meaning(action))
            obs, reward, done, info = env.step(action)
            j+=1
            if info['enemy_health']==0:
                win+=1

    print('Total win out of 10 is ',win)


practise()

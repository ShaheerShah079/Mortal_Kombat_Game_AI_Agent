import os
import shutil
import pandas as pd

df=pd.read_csv('MK2/N/Dataset.csv')
images=df['Images']
decimal_actions=df['decimal Actions']
alpha_actions=df['Alpha Action']

dataframe=pd.read_csv('125_Alpha2Bin.csv')
alpha=dataframe['Alpha']
decimal=dataframe['Decimal']
classes=dataframe['class']

aplha_to_class={}
for i in range(125):
    aplha_to_class[alpha[i]]=classes[i] 

rootdir='Classes_Dataset'
# os.mkdir(rootdir)
#
# for c in classes:
#     os.mkdir(os.path.join(rootdir,str(c)))

try:
    for i in range(len(images)):
        shutil.copy(images[i], os.path.join(rootdir, str(aplha_to_class[alpha_actions[i]])))
        print(i)
except Exception as e:
    print(e)

import tensorflow as tf
import os
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten,Dropout, BatchNormalization
import pandas as pd
import matplotlib.pyplot as plt
import  numpy as np


# df=pd.read_csv('MK2/N/Dataset.csv')
# images=df['Images']
# decimal_actions=df['decimal Actions']
# alpha_actions=df['Alpha Action']
#
# dataframe=pd.read_csv('125_Alpha2Bin.csv')
# alpha=dataframe['Alpha']
# decimal=dataframe['Decimal']
# classes=dataframe['class']
#
# aplha_to_class={}
# for i in range(125):
#     aplha_to_class[alpha[i]]=classes[i]
#
# labels=[]
# for i in range(len(images)):
#     labels.append(aplha_to_class[alpha_actions[i]])

rootdir = 'Dataset'
data = tf.keras.utils.image_dataset_from_directory(
    os.path.join(rootdir),
    image_size=(224, 320),
    batch_size=32,
    shuffle=True,
)

data=data.map(lambda images,lab : (images/255,lab))

data = data.map(lambda x, y: (x, tf.one_hot(y,125)))

train_size=int(len(data)*.70)
val_size=int(len(data)*.15)
test_size=int(len(data)*.15)

train=data.take(train_size)
validation=data.skip(train_size).take(val_size)
testing_data=data.skip(train_size+val_size).take(test_size)

model = Sequential()
model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(224,320,3)))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.5))
model.add(Dense(125, activation='softmax'))
model.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'],run_eagerly=True)

history = model.fit(train, validation_data=validation, epochs=3)

# Evaluate the model on the testing data
test_loss, test_accuracy = model.evaluate(testing_data)

# Print the testing accuracy
print(f'Testing Accuracy: {test_accuracy}')

# Plot the training and validation accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


# model.save(os.path.join('models','N_13k_NEW_CNN.keras'))

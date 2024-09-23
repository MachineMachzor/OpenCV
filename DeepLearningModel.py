# 1. Want a couple thousand images of each class optimally
# 2. OpenCV just to open an image and resize the model
import os
import caer 
import canaro 
import numpy as np 
import cv2 as cv 
import gc
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler


simpsonsLoc = "D:\Simpsons"
simpsonsDataSet = fr"{simpsonsLoc}\simpsons_dataset"

#Should be standardized
IMG_SIZE = (80,80) #This size works well for the model
channels = 1 #Grayscale, we do not require colors
char_path = simpsonsDataSet

char_dict = {}
for char in os.listdir(char_path):
    char_dict[char] = len(os.listdir(f"{char_path}\{char}"))

#Sort the dictionary in descending order
char_dict = caer.sort_dict(char_dict, descending=True) #Largest to smallest

characters = []
count = 0
for i in char_dict:
    characters.append(i[0])
    count += 1
    if count >= 10:
        break

#Create the training data
#Goes through images inside folder and adds it to the training set
train = caer.preprocess_from_dir(char_path, characters, channels=channels, IMG_SIZE=IMG_SIZE, isShuffle=True)

# len(train)
plt.figure(figsize=(30,30)) #OpenCV does not display properly in jupyter notebook
plt.imshow(train[0][0], cmap='gray') #First element in training set
plt.show()


#Seperate the training set into features and labels
#Features are the images and labels are the names of the characters
featureSet, labels = caer.sep_train(train, IMG_SIZE=IMG_SIZE)

# Normalize the featureSet between (0,1)
featureSet = caer.normalize(featureSet)
labels = to_categorical(labels, len(characters)) #len(characters) is number of categories

#Split the training data into training and validation sets
x_train, x_val, y_train, y_val = caer.train_val_split(featureSet, labels, val_ratio=.2) #20% validation

#Delete the training data to save memory
del train
del featureSet
del labels
gc.collect()

BATCH_SIZE = 32 
EPOCHS = 10

# Image data generator
datagen = canaro.generators.imageDataGenerator()
train_gen = datagen.flow(x_train, y_train, batch_size=32)



# Create our model
model = canaro.models.createSimpsonsModel(IMG_SIZE=IMG_SIZE, channels=channels, output_dim=len(characters), 
                                          loss='binary_crossentropy', decay=1e-6, learning_rate=0.001, momentum=0.9, nesterov=True)

# model.summary()

# Callbacks list
# Schedule the learning rate to train better
# callbacks_list = [tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=3),
#                   tf.keras.callbacks.ModelCheckpoint(filepath='model_checkpoint.h5', monitor='val_loss', save_best_only=True),
#                   tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2)]
callbacks_list = [LearningRateScheduler(canaro.lr_schedule)]
training = model.fit(train_gen,
                     steps_per_epoch=len(x_train)//BATCH_SIZE,
                     epochs=EPOCHS,
                     validation_data=(x_val, y_val),
                     validation_steps=len(y_val)//BATCH_SIZE,
                     callbacks=callbacks_list)
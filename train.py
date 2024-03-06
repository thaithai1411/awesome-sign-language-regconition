import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Activation


DIRECTORY = r"datas"
CATEGORIES =['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

data = []

for categories in CATEGORIES:
    folder = os.path.join(DIRECTORY, categories)
    label = CATEGORIES.index(categories)
    # print(folder)

    for img in os.listdir(folder):
        img = os.path.join(folder, img)
        # print(img)
        img_arr = cv2.imread(img)
        # print(img_arr)
        # plt.imshow(img_arr)
        img_arr = cv2.resize(img_arr,(224,224))
        # plt.imshow(img_arr)
        # break

        data.append([img_arr,label])
        
# print(data)

random.shuffle(data)

x = []
y = []
for features, label in data:
    x.append(features)
    # print(x)
    y.append(label)
    # print(features)

x = np.array(x)
y = np.array(y)
x = x/255
# y = y/255
# print(y.shape)

model=Sequential()
model.add( Conv2D(64,(3,3),input_shape=x.shape[1:],activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add( Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add( Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(26,activation='softmax'))
model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy' ,metrics=['accuracy'])
model_fit = model.fit(x, y, epochs=5, batch_size= 128, validation_split=0.1)

model.save('models/mymodel_18_7_17h43.h5')

# plt.plot(model_fit.history['accuracy'])
# plt.plot(model_fit.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc = 'upper left')
# plt.show()

# plt.plot(model_fit.history['loss'])
# plt.plot(model_fit.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc = 'upper left')
# plt.show()
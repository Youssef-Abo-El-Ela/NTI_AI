# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 15:42:35 2023

@author: Youssef Aboelela
"""
import keras
import tensorflow as tf
from keras import activations,optimizers,layers,datasets,utils

from keras.datasets import mnist

((X_train,y_train),(X_test,y_test))=mnist.load_data()

X_train=X_train.astype('float32')/255
X_test=X_test.astype('float32')/255

train_X=X_train.reshape(-1,28,28,1)
test_X=X_test.reshape(-1,28,28,1)

train_y=keras.utils.to_categorical(y_train)
test_y=keras.utils.to_categorical(y_test)

model=keras.Sequential()
model.add(layers.Conv2D(filters=32, kernel_size=5 , strides=1,padding='same',activation=tf.nn.relu,input_shape=(28,28,1)))
model.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2),padding='valid'))

model.add(layers.Conv2D(filters=64, kernel_size=3,padding='same',activation=tf.nn.relu,strides=1))
model.add(layers.MaxPool2D(pool_size=(2,2),padding='valid',strides=(2,2)))

model.add(layers.Dropout(0.25))

model.add(layers.Flatten())

model.add(layers.Dense(units=128,activation=tf.nn.relu))
model.add(layers.Dropout(0.5))

model.add(layers.Dense(units=10,activation=tf.nn.softmax))

model.compile(optimizer='adam',metrics='accuracy',loss='categorical_crossentropy')

history=model.fit(train_X,train_y,batch_size=128,epochs=10)

score=model.evaluate(test_X,test_y,verbose=1)

print('Accuracy of evaluation is ',score[1])

import matplotlib.pyplot as plt

accuracy=history.history['accuracy']
plt.plot(accuracy,'g',label='Training Accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epoches')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

loss=history.history['loss']
plt.plot(loss,'g',label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epoches')
plt.ylabel('Loss')
plt.legend()
plt.show()

model.summary()
#utils.plot_model(model, to_file='model_visualization.png', show_shapes=True)
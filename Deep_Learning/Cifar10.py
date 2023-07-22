# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 16:22:01 2023

@author: Youssef Aboelela
"""

from keras import utils,optimizers,layers,Sequential,datasets
import tensorflow as tf
import keras
import matplotlib.pyplot as plt

((X_train,y_train),(X_test,y_test)) = tf.keras.datasets.cifar10.load_data()

train_X=X_train.astype('float32')/255
test_X=X_test.astype('float32')/255

train_y=keras.utils.to_categorical(y_train)
test_y=keras.utils.to_categorical(y_test)

model=Sequential()
model.add(layers.Conv2D(filters=32,kernel_size=5,activation=tf.nn.relu,strides=1,padding='same',input_shape=(32,32,3)))
model.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2),padding='valid'))

model.add(layers.Conv2D(filters=64,kernel_size=3,activation=tf.nn.relu,strides=1,padding='same'))
model.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2),padding='valid'))

model.add(layers.Dropout(0.25))
model.add(layers.Flatten())

model.add(layers.Dense(units=128,activation=tf.nn.relu))
model.add(layers.Dropout(rate=0.5))

model.add(layers.Dense(units=10,activation=tf.nn.softmax))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics='accuracy')

history=model.fit(train_X,train_y,batch_size=128,epochs=10)

score=model.evaluate(test_X,test_y,verbose=1)
print('Accuracy of Evaluation = ', score[1])

accuracy=history.history['accuracy']
plt.plot(accuracy,'g',label='Cifar10')
plt.xlabel('Epoches')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

accuracy=history.history['loss']
plt.plot(accuracy,'g',label='Cifar10')
plt.xlabel('Epoches')
plt.ylabel('Loss')
plt.legend()
plt.show()

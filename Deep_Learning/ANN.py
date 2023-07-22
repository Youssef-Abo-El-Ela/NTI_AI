# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from keras.layers import Dense
from keras.models import Sequential
import pandas as pd
import matplotlib.pyplot as plt


dataset=pd.read_csv('diabetes3.csv')
X=dataset.iloc[:,0:8].values
y=dataset.iloc[:,8].values

from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=0,strategy='mean')
X_correct=imputer.fit_transform(X)

model=Sequential()
model.add(Dense(24,input_dim=8,activation='relu'))
model.add(Dense(18,activation='relu'))
model.add(Dense(12,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='sigmoid')) #Sigmoid for binary outcomes and Softmax for multiclass outcome

model.compile(loss="binary_crossentropy",optimizer='adam',metrics='accuracy')

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X_correct,y,test_size=0.3)

history=model.fit(X_train,y_train,batch_size=10,epochs=150)

scores=model.evaluate(X_test,y_test)
print('Accuracy = ',scores[1]*100)

from ann_visualizer.visualize import ann_viz
ann_viz(model,title='My First Neural Netwrok',filename='My First Neural Netwrok')

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
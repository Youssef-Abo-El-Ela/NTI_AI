# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 13:06:20 2023

@author: Youssef Aboelela
"""

import numpy as np
import pandas as pd
from  sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('Missing_Data.csv')
X=dataset.iloc[:,:-1].values
col2=dataset.iloc[:,[1]].values
col3=dataset.iloc[:,[2]].values

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

imputer.fit(col2)
col2=imputer.transform(col2)
imputer.fit(col3)
col3=imputer.transform(col3)

dataset.iloc[:,[1]]=col2
dataset.iloc[:,[2]]=col3
X=dataset.iloc[:,:-1].values

ct=ColumnTransformer([('Country', OneHotEncoder(),[0])], remainder="passthrough")
X=ct.fit_transform(X)

mapping={'Yes':1,'No':0}
dataset=dataset.applymap(mapping.get)
X=X[:,1:]
y=dataset.iloc[:,-1].values

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.5)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)
accuracy=(cm[0][0]+cm[1][1])/(len(y_test))
print(accuracy)






'''
OLD WAY (NOT WORKING)
lb_enc=LabelEncoder()
X[:,0]=lb_enc.fit_transform(X[:,0])

one_hot=OneHotEncoder()
X=one_hot.fit_transform(X)
X=pd.DataFrame(X)
'''
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 15:39:31 2023

@author: Youssef Aboelela
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


dataset=pd.read_csv('diabetes3.csv')
X=dataset.iloc[:,0:8].values
y=dataset.iloc[:,8].values

imputer=SimpleImputer(missing_values=0,strategy='mean')
X=imputer.fit_transform(X)

sc=StandardScaler()
X=sc.fit_transform(X)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

#Decision Tree Algortihm


tree=DecisionTreeClassifier(criterion='entropy')
tree.fit(X_train, y_train)

y_pred_tree=tree.predict(X_test)

cm_tree=confusion_matrix(y_test, y_pred_tree)
acc_tree=(cm_tree[0][0]+cm_tree[1][1])/len(y_test)
precision_tree=cm_tree[1][1]/(cm_tree[1][1]+cm_tree[0][1])
true_pos_rate_tree=cm_tree[1][1]/(cm_tree[1][1]+cm_tree[1][0])
true_neg_rate_tree=cm_tree[0][0]/(cm_tree[0][0]+cm_tree[0][1])

print('Decision Tree:')
print('Accuracy= ',acc_tree*100)
print('Precision=', precision_tree*100)
print('True Positive Rate=',true_pos_rate_tree*100)
print('True Negative Rate= ',true_neg_rate_tree*100)

#Random Forest Algorithm


rand_forest=RandomForestClassifier(n_estimators=10 , criterion='entropy')
rand_forest.fit(X_train, y_train)
y_pred_forest=rand_forest.predict(X_test)

cm_forest=confusion_matrix(y_test, y_pred_forest)
acc_forest=(cm_forest[0][0]+cm_forest[1][1])/len(y_test)
precision_forest=cm_forest[1][1]/(cm_forest[1][1]+cm_forest[0][1])
true_pos_rate_forest=cm_forest[1][1]/(cm_forest[1][1]+cm_forest[1][0])
true_neg_rate_forest=cm_forest[0][0]/(cm_forest[0][0]+cm_forest[0][1])

print('Random Forest:')
print('Accuracy: ',acc_forest*100)
print('Precision=', precision_forest*100)
print('True Positive Rate=',true_pos_rate_forest*100)
print('True Negative Rate= ',true_neg_rate_forest*100)

log_reg=LogisticRegression(random_state=0)
log_reg.fit(X_train, y_train)

y_pred_log_reg=log_reg.predict(X_test)

cm_log_reg=confusion_matrix(y_test, y_pred_log_reg)
acc_log_reg=(cm_log_reg[0][0]+cm_log_reg[1][1])/(len(y_test))
precision_log_reg=cm_log_reg[1][1]/(cm_log_reg[1][1]+cm_log_reg[0][1])
true_pos_rate_log_reg=cm_log_reg[1][1]/(cm_log_reg[1][1]+cm_log_reg[1][0])
true_neg_rate_log_reg=cm_log_reg[0][0]/(cm_log_reg[0][0]+cm_log_reg[0][1])

print('Logistic Regression:')
print('Accuracy: ',acc_log_reg*100)
print('Precision=', precision_log_reg*100)
print('True Positive Rate=',true_pos_rate_log_reg*100)
print('True Negative Rate= ',true_neg_rate_log_reg*100)


#SVM

from sklearn.svm import SVC
svc=SVC(kernel='linear',random_state=10)
svc.fit(X_train, y_train)

y_pred_svm=svc.predict(X_test)

cm_svc=confusion_matrix(y_test, y_pred_svm)
acc_svc=(cm_svc[0][0]+cm_svc[1][1])/(len(y_test))
precision_svc=cm_svc[1][1]/(cm_svc[1][1]+cm_svc[0][1])
true_pos_rate_svc=cm_svc[1][1]/(cm_svc[1][1]+cm_svc[1][0])
true_neg_rate_svc=cm_svc[0][0]/(cm_svc[0][0]+cm_svc[0][1])

print('SVC:')
print('Accuracy: ',acc_svc*100)
print('Precision=', precision_svc*100)
print('True Positive Rate=',true_pos_rate_svc*100)
print('True Negative Rate= ',true_neg_rate_svc*100)


#KNN

knn=KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)

y_pred_knn=knn.predict(X_test)

cm_knn=confusion_matrix(y_test, y_pred_knn)
acc_knn=(cm_knn[0][0]+cm_knn[1][1])/(len(y_test))
precision_knn=cm_knn[1][1]/(cm_knn[1][1]+cm_knn[0][1])
true_pos_rate_knn=cm_knn[1][1]/(cm_knn[1][1]+cm_knn[1][0])
true_neg_rate_knn=cm_knn[0][0]/(cm_knn[0][0]+cm_knn[0][1])

print('KNN:')
print('Accuracy: ',acc_knn*100)
print('Precision=', precision_knn*100)
print('True Positive Rate=',true_pos_rate_knn*100)
print('True Negative Rate= ',true_neg_rate_knn*100)

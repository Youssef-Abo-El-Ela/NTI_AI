# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 20:56:55 2023

@author: Youssef Aboelela
"""
import pandas as pd
import matplotlib.pyplot as plt

#Sorting and Removing Duplicates
dataset=pd.read_csv('Wuzzuf_jobs.csv')

describtion=dataset.describe()
dataset.info()

dataset.sort_values('Title',inplace=True)

dataset.drop_duplicates(keep = 'first' , inplace = True)

#Bar and Pie Representation

numofJobs=dataset['Company'].value_counts().head(5)

#max_length=10

#truncated_list = [s[:max_length] for s in numofJobs]

labels=list(numofJobs.keys())
'''
plt.pie(numofJobs, labels=labels)
plt.figure(figsize=(15,15))
plt.xticks(rotation=90)
plt.bar(labels,numofJobs)
'''
#Skills

skills = []
skill=[]
for row in dataset['Skills']:  
    skills.append(row.split(','))

for row in skills:
    for element in row:
        skill.append(element)

skill_dataframe=pd.DataFrame(skill)
skill_describe = skill_dataframe.describe()

print('The most required skill is',skill_dataframe[0].mode()[0] )
skill=list(set(skill))
skill.sort()

#Factorize

factorized_YOE , guide = pd.factorize(dataset['YearsExp'])
factorized_YOE_frame = pd.DataFrame(factorized_YOE)
most_frequent_YOE = guide[factorized_YOE_frame[0].mode()[0]]
print('The most frequent Years of Experience is ',most_frequent_YOE)

#KMeans
from sklearn.cluster import KMeans

X_required=dataset.iloc[:,[0,1]]
X_required['Title'],guide_title=pd.factorize(X_required['Title'])
X_required['Company'],guide_company=pd.factorize(X_required['Company'])
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X_required)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X_required)
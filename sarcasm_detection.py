#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 14:22:07 2023

@author: Sarvandani
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
data = pd.read_json("Sarcasm.json", lines=True)
print(data.head())
print(data.isnull().sum())

#--------------------------------
# writing data with lable and feature
#feature column: headline
#lable column:is_sarcastic 
data = data[["headline", "is_sarcastic"]]
#splitting data set or seperation between lable and feature
# feature
x = np.array(data["headline"])
#lable
y = np.array(data["is_sarcastic"])
#--------------------------------
# simple language: Text should be converted into the numbers!
countv=CountVectorizer()
x = countv.fit_transform(x)
Xtrain, Xtest, Ytrain, Ytest = train_test_split(x, y, test_size=0.33, random_state=7)
#_________________________________
#Machin learning: define models and fiiting model
model1 = BernoulliNB()
model1.fit(Xtrain, Ytrain)
#____________________________
model2 = DecisionTreeClassifier()
model2.fit(Xtrain, Ytrain)
#________________________________
model3 = LogisticRegression(solver='lbfgs',max_iter=2000,random_state = 0)
model3.fit(Xtrain, Ytrain)
#_____________________________
#_____________________________
model4 = KNeighborsClassifier(algorithm='auto', 
                           leaf_size=30, 
                           metric='minkowski',
                           p=2,
                           metric_params=None, 
                           n_jobs=1, 
                           n_neighbors=5, 
                           weights='uniform')
model4.fit(Xtrain, Ytrain)
#__________________
#detection with different models
statement = "christian bale visits sikh temple victims"
statement_converted = countv.transform([statement]).toarray()
Decetion1 = print(model1.predict(statement_converted))
#______________________________
Decetion2 = print(model2.predict(statement_converted))
#_____________________________________
Decetion3 = print(model3.predict(statement_converted))
#_________________________________________________
Decetion4 = print(model4.predict(statement_converted))


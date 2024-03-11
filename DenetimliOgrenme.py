# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 22:22:03 2024

@author: fatih
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn import datasets
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB



#İrirs veri seti dahil edilmesi

from os import X_OK
iris = datasets.load_iris()
X = iris.data
Y = iris.target



#Splitting X and Y into Training and Testing sets

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=1)

dt = DecisionTreeClassifier(random_state=0)
dt.fit(X_train, Y_train)
dt_pred = dt.predict(X_test)


print("Accuracy Score of DT Classifier: ", accuracy_score(Y_test, dt_pred))
print("\nPrecision Score of DT Classifier: ", precision_score(Y_test, dt_pred, average="weighted"))
print("\nRecall Score of DT Classifier: ", recall_score(Y_test, dt_pred, average="weighted"))
print("\nF1 Score of DT Classifier: ", f1_score(Y_test, dt_pred, average="weighted"))
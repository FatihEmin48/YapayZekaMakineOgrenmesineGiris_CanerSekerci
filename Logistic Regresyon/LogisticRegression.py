# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 13:41:30 2024

@author: fatih
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


X,y = make_classification(n_samples=1000, n_features=5, n_classes=2, random_state=42)


#Veri setini eğitim ve test setine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#Logistic Regresyon modelinin oluşturulması
model = LogisticRegression()
model.fit(X_train, y_train)


#Test seti üzerine modelin doğruluğunu değerlendirme
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Modelin Doğruluğu: ", accuracy)


#Confision Matrix Yazdırma
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confision Matrix: ")
print(conf_matrix)
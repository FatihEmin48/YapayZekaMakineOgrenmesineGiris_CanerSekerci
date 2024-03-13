# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 18:47:30 2024

@author: fatih
"""

from sklearn.datasets import load_breast_cancer
import pandas as pd


#Veri Seti Yükleme
cancer = load_breast_cancer()



#Veri Seti Dönüştürme
data = pd.DataFrame(data = cancer.data, columns = cancer.feature_names)
data["target"] = cancer.target

print(data.head())



from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


X = data.drop("target", axis=1)
y = data["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LogisticRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Model Doğruluğu: ", accuracy)




import matplotlib.pyplot as plt

plt.bar(["Accuracy"], [accuracy])
plt.ylim(0,1)
plt.ylabel("Accuracy")
plt.title("Model Doğruluğu")
plt.show()




train_size = len(X_train)
test_size = len(X_test)

plt.bar(["Eğitim Verisi", "Test Verisi"], [train_size, test_size])
plt.ylabel("Veri Sayısı")
plt.title("Eğitim ve Test Verisi Boyutları")
plt.show()
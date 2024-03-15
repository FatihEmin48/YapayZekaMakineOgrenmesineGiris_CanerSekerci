# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 14:03:14 2024

@author: fatih
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


data ={"Yas":[22, 25, 30, 35, 40, 45, 50, 55, 60, 65],
       "Sigara":[0, 0, 1, 0, 1, 1, 1, 0, 1, 1],
       "Hipertansiyon":[0, 0, 1, 1, 1, 0, 1, 1, 1, 0],
       "Koroner_arter_hastalığı":[0, 0, 1, 1, 1, 0, 1, 1, 1, 0]}

df = pd.DataFrame(data)


X = df[["Yas", "Sigara", "Hipertansiyon"]]
y = df["Koroner_arter_hastalığı"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Model Doğruluğu:", accuracy)
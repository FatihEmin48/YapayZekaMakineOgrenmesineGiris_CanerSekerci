# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 09:41:13 2024

@author: fatih
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree


np.random.seed(0) #Rastgele sayı üreticini başlatma
age = np.random.randint(20, 80, 30)
blood_pressure = np.random.randint(80, 180, 30)
cholestrol = np.random.randint(120, 250, 30)
disease = np.random.randint(0, 2, 30) #0:Hastalıksız, 1:HAstalıklı

data = pd.DataFrame({"Yaş":age, "KanBasıncı":blood_pressure, "Kolesterol":cholestrol, "Hasta":disease})

X = data.drop("Hasta", axis=1)
y = data["Hasta"]

rf_classifier = RandomForestClassifier(n_estimators=5, random_state=42)
rf_classifier.fit(X, y)


plt.figure(figsize=(20,10))
tree.plot_tree(rf_classifier.estimators_[0], filled=True, feature_names=X.columns, class_names=["Sağlıklı","Hasta"], 
               fontsize=10, rounded=True, precision=2, label="all", impurity=True, proportion=True)
plt.show()
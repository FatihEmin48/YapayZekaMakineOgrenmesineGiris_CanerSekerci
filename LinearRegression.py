# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 18:07:15 2024

@author: fatih
"""

import numpy as np

X = np.array([50,100,150,200,250])
y = np.array([100000,200000,300000,400000,500000])



import matplotlib.pyplot as plt

plt.scatter(X, y)
plt.xlabel("Ev metrekare")
plt.ylabel("Ev Fiyatı(₺)")
plt.title("Ev Metrekarelerine Göre Fiyatlar")
plt.show() 




from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X.reshape(-1,1), y)

ev_metrekare = 120
tahmin_fiyat = model.predict([[ev_metrekare]])
print(f"{ev_metrekare} metrekarelik bir evin tahmin fiyatı: {tahmin_fiyat[0]:.2f}")
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN



#Verilerin oluşturulması
X, _ = make_moons(n_samples=200, noise=0.1, random_state=42)



#DBSCAN => Veriyi yoğunluğa göre kümeliyor
dbscan = DBSCAN(eps=0.2, min_samples=5)
cluster = dbscan.fit_predict(X)

n_clusters = len(set(cluster)) - (1 if -1 in cluster else 0)
n_noise = list(cluster).count(-1)



#Görselleştirme
plt.figure(figsize=(8, 6))
plt.scatter(X[cluster == -1,0], X[cluster == -1,1], c="gray", marker="o", s=30, label="Gürültü")

for i in range(n_clusters):
    plt.scatter(X[cluster == i,0], X [cluster == i,1], label= f"Küme{i+1}")
    
plt.title("DBSCAN ile Kümeleme")
plt.xlabel("Özellik 1")
plt.ylabel("Özellik 2")
plt.legend()
plt.show()
          
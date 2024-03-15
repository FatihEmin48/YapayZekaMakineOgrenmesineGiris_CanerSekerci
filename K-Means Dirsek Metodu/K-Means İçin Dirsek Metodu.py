# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 21:02:51 2024

@author: fatih
"""

#K-Means Algoritmasında Optimum K Değeri İçin Dirsek Metodu Uygulaması => Düzleşmenin başladığı nokta


import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from scipy.spatial.distance import cdist

X, _ = make_blobs(n_samples=1000, centers=5, cluster_std=1.0, random_state=42)

distotions = []
inertias = []

K = range(1, 10)

for k in K:
    kmeanModel = KMeans(n_clusters=k, n_init=10).fit(X) #İlk eğitimi gerçekleştirme
    kmeanModel.fit(X) #Kümelre ayırıyoruz
    
    distotions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, "euclidean"), axis=1))/X.shape[0])
    inertias.append(kmeanModel.inertia_)
    
    
    
#Görselleştirme

#Dirsek Metodu Grafiği
plt.figure(figsize=(8, 6))
plt.plot(K, distotions, "bx-")
plt.xlabel("Küme Sayısı")
plt.ylabel("Çarpıklık(Distortions)")
plt.title("Dirsek Yöntemi  -  Çarpıklık")
plt.show()


#Inertia Grafiği
plt.figure(figsize=(8, 6))
plt.plot(K, inertias, "bx-")
plt.xlabel("Küme Sayısı")
plt.ylabel("Toplam Hata(Inertia")
plt.title("Dirsek Yöntemi(Inertia)")
plt.show()


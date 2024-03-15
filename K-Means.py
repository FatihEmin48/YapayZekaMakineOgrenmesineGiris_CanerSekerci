# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 17:17:24 2024

@author: fatih
"""

import numpy as np
import matplotlib.pyplot as plt

#Verilerin Oluşturulması
np.random.seed(42)
X = np.random.rand(100, 2)


#K-Means Algoritması Oluşturulması
def k_means(X, k, max_iters=100):
    #Rastgele K Merkez Seçimi
    centroids = X[np.random.choice(range(len(X)), size=k, replace=False)] 
    
    for _ in range(max_iters):
        #Her Noktaya En Yakın Merkezin İndekleri
        labels = np.argmin(np.linalg.norm(X[:,np.newaxis]-centroids,axis=2), axis=1)
        
        
        #Yeni Merkezlerin Hesaplanması
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)]) 
        
        
        #Merkezlerde Bir Değişiklik Yoksa Çıkış Yapma
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
        
    return centroids, labels


#K-Means Uygulama
k = 3
centroids, labels = k_means(X, k)


#Sonuçları Görselleştirme
plt.scatter(X[:,0], X[:,1], c=labels, cmap="viridis", alpha=0.5)
plt.scatter(centroids[:,0], centroids[:,1], c="red", marker="x") #Kümelerin merkezini gösteriyor
plt.title("K-Means Kümeleme")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

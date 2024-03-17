# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 20:40:28 2024

@author: fatih
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import SpectralClustering


X, _ = make_moons(n_samples=200, noise=0.1, random_state=42)



spectral_clustering = SpectralClustering(n_clusters=2, affinity="nearest_neighbors", random_state=42)
clusters = spectral_clustering.fit_predict(X)


plt.figure(figsize=(8, 6))
plt.scatter(X[:,0], X[:,1], c=clusters, cmap="viridis")
plt.title("Spektral Kümeleme")
plt.xlabel("Özellik 1")
plt.ylabel("Özellik 2")
plt.colorbar()
plt.show()

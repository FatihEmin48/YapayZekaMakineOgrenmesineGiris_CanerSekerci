# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 15:06:08 2024

@author: fatih
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs


X, y = make_blobs(n_samples=50, centers=2, random_state=6)


clf = svm.SVC(kernel="linear")

clf.fit(X,y)



#Eğitilmiş sınıfın karar sınırlarını görselleştirme
plt.scatter(X[:,0], X[:,1], c=y, s=30, cmap=plt.cm.Paired)


#Karar sınırlarını çizme
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()


#Modelin karar sınırlarını oluşturma
xx = np.linspace(xlim[0], xlim[1],30)
yy = np.linspace(ylim[0], ylim[1],30)

YY, XX = np.meshgrid(yy, xx)

xy = np.vstack([XX.ravel(), YY.ravel()]).T

Z = clf.decision_function(xy).reshape(XX.shape)


#Karar sınırlarını ve destek vektörlerini çizme
ax.contour(XX, YY, Z, colors = "k", levels=[-1,0,1], alpha = 0.5, linestyles = ["--", "-", "--"])
#Destek vektörleri görselleştirme adımı
ax.scatter(clf.support_vectors_[:,0], clf.support_vectors_[:,1],s=100,linewidth=1,facecolor="none",edgecolor="k")



plt.show()
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 15:38:01 2024

@author: fatih
"""

from sklearn.tree import DecisionTreeClassifier
import pandas as pd

veri ={
       "Hava Durumu":["Güneşli", "Güneşli", "Bulutlu", "Yağmurlu", "Yağmurlu", "Bulutlu", "Güneşli", "Güneşli", "Bulutlu", "Yağmurlu"],
       "Sıcaklık":["Sıcak","Sıcak","Sıcak","Ilık","Soğuk","Soğuk","Soğuk","Ilık","Ilık","Ilık"],
       "Rüzgar":["Rüzgarsız","Rüzgarlı","Rüzgarsız","Rüzgarsız","Rüzgarsız","Rüzgarlı","Rüzgarsız","Rüzgarlı","Rüzgarsız","Rüzgarlı"],
       "Gitme":["E","E","E","E","H","E","E","H","E","H"]
       }

df = pd.DataFrame(veri)


#Kategorik verileri sayısal verilere dönüştürme
df["Hava Durumu"] = df["Hava Durumu"].map({"Güneşli":0, "Bulutlu":1, "Yağmurlu":2})
df["Sıcaklık"] = df["Sıcaklık"].map({"Sıcak":0, "Ilık":1, "Soğuk":2})
df["Rüzgar"] = df["Rüzgar"].map({"Rüzgarsız":0, "Rüzgarlı":1})
df["Gitme"] = df["Gitme"].map({"H":0, "E":1})


#Özellikler ve hedef değişkeni ayırma
X = df.drop("Gitme", axis=1)
y = df["Gitme"]


#Model oluşturma
dt_model = DecisionTreeClassifier()


#Model eğitim
dt_model.fit(X,y)


#Yeni veri örneği
yeni_veri = pd.DataFrame({"Hava Durumu":["Bulutlu"], "Sıcaklık":["Ilık"], "Rüzgar":["Rüzgarlı"]})
yeni_veri["Hava Durumu"] = yeni_veri["Hava Durumu"].map({"Güneşli":0, "Bulutlu":1, "Yağmurlu":2})
yeni_veri["Sıcaklık"] = yeni_veri["Sıcaklık"].map({"Sıcak":0, "Ilık":1, "Soğuk":2})
yeni_veri["Rüzgar"] = yeni_veri["Rüzgar"].map({"Rüzgarsız":0, "Rüzgarlı":1})


#Tahmin
tahmin = dt_model.predict(yeni_veri)


if tahmin[0]==1:
    print("Restorana Git: (E)")
else:
    print("Restorana Gitme: (H)")
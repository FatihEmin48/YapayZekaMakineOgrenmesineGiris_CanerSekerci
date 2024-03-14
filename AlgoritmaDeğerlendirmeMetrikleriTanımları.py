# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 11:22:47 2024

@author: fatih
"""

#TP: Modelin doğru bir şekilde bir sınıfı doğru tahmin ettiği durumların sayısıdır.
#FP: Modelin yanlış bir şekilde bir sınıfın pozitif olarak tahmin ettiği durumların sayısıdr.
#TN: Modelin doğru bir şekilde bir sınıfı negatif olrak tahmin ettiği durumların sayısıdır.
#FN: Modelin yanlış bir şekilde bir sınıfı negatif olrak tahmin ettiği durumların sayısıdır.


# Accuracy Score (Doğruluk): (TP + TN) / (TP + FP + TN + FN) = Doğruluk modelin doğru tahmin edilen örneklerin tüm örnek sayısına oranını ifade eder.

# Precision (Kesinlik): TP / (TP + FP) = Modelin pozitif olarak tahmin ettiği örneklerin gerçekten pozitif olma oranı.

# Recall (Duyarlılık): TP / (TP + FN) = Duyarlılık gerçekten pozitif olan örneklerin ne kadarının model tarafından tespit edildiğinin oranını ifade eder.

# F1 Score:  2 * (Precision * Recall) / (Precision + Recall) = Precision ve Recal değerlerinin hormanik ortalamasıdır. Modelin sınıflandırma performansını dengelemek için kullanılır

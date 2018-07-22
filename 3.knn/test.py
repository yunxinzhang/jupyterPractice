# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 03:37:34 2018

@author: zyx
"""
# ？ 因为完全依赖于距离的计算，对于维度大的数据，有维度灾难的问题。 百万级别的维度会不能处理。
# =============================================================================
from data_split import data_split
from data_split import calc_ac
from knnmy import KNNCLF
from scaling import scaling
# =============================================================================
from sklearn import datasets

#test KNNCLF

iris = datasets.load_iris()

# seed 固定， 便于测试随机数
# seed = 123 时 scaling 效果差， 456 效果好
Xtr, Ytr, Xt, Yt = data_split(iris.data, iris.target.reshape(-1,1), seed=123)
sc = scaling()
sc.fit(Xtr)
Xtr1 = sc.transform(Xtr)
Xtr1 = Xtr
knnmy = KNNCLF()

knnmy.fit(Xtr1,Ytr)
Xt1 = sc.transform(Xt)
Xt1 = Xt
y_pred = knnmy.predict2all(Xt1)

ac1 = calc_ac(y_pred, Yt)

#test sklearn

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
sc.fit(Xtr)
Xtr2 = sc.transform(Xtr)
#Xtr2 = Xtr
print(Ytr.shape)
Ytr.reshape(Ytr.shape[0],)
knn.fit(Xtr2, Ytr)
Xt2 = sc.transform(Xt)
#Xt2 = Xt
y_pred2 = knn.predict(Xt2)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
ac = accuracy_score(Yt, y_pred2)
cm = confusion_matrix(Yt, y_pred2)


# test sklearn all
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
Xtr3 =  ss.fit_transform(Xtr)
knn.fit(Xtr3, Ytr)
Xt3 = ss.transform(Xt)
y_pred3 = knn.predict(Xt3)
ac3 = accuracy_score(Yt, y_pred3)




# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 07:06:09 2018

@author: zyx
"""
import numpy as np
from collections import Counter
# 函数的实现可以是 复杂的，但是调用必须 简单。设计函数的原则。 暂缓考虑这些细节。

class KNNCLF():
    def __init__(self):
        pass
    
        
    def fit(self ,X, Y, k=5):
        assert X.shape[0] == Y.shape[0], "shape must match"
        assert k > 0 and k < X.shape[0], "k must > 0 and x.shape[0] must > k"
        self.X_train = X
        self.Y_train = Y
        self.k = k
        return self
    
    def getDist(self, i, Xp):
        sz = np.shape(self.X_train)[1]
       # print("sz", sz)
        dist = 0 
        for j in range(sz):
            dist += (self.X_train[i][j]-Xp[0][j])**2
        return dist
    
    def predict(self , Xp):
        dists = [self.getDist(i, Xp) for i in range(np.shape(self.X_train)[0])]
   #    print(dists)
        distinds = np.argsort(np.array(dists))
   #     print(distinds)
        topN = [self.Y_train[i][0] for i in distinds]
   #     print(topN)
        voter = Counter(topN[0:3])
   #     print(voter)
        return voter.most_common(1)[0][0]
    
    def predict2(self , xp):
        # y_train 统一格式， 我的方式。 关系不大。 对于坐标值要什么要清晰
        assert xp.shape[1] == self.X_train.shape[1], "shape must match"
        # np.array 的 ** 运算含义， sum 含义
        dists = [np.sqrt(np.sum((xp-xt)**2)) for xt in self.X_train ]
        # argsort 可以直接排序 py[] ， 自动转型
        # 命名要语义清晰
        knearest = np.argsort(dists)
        # 只取有用数量的
        topK = [self.Y_train[i][0] for i in knearest[0:self.k]]
        voter = Counter(topK)
        #print(type(voter))  #<class 'collections.Counter'>
        #print(voter)        # Counter({0: 4, 1: 1})
        # most_common(n) 返回数量最多的前 n 个元素
        return voter.most_common(1)[0][0]
    def predict2all(self, xp):
        return [self.predict2(x.reshape(1,-1)) for x in xp]
# __name__ ==  __main__ 
if __name__ == "__main__":
    xt = np.array([1,0,1.1,0,1.2,0,1.11,0,
                   2,1,2.1,1,2.2,1,2.12,1]).reshape(-1,2)
    yt = np.array([0,      0,    0,     0,
                   1,      1,    1,     1]).reshape(-1,1)
    
    knn= KNNCLF()
    knn.fit(xt,yt)
    xp= np.array([1.21, 0]).reshape(1,-1)
    cls = knn.predict2(xp)
    print(cls)
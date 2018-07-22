# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 12:30:46 2018

@author: zyx
"""
# 没有 scaling 的话， 会导致 数值大 的特征成为分类的主导。
import numpy as np

class scaling():
    
    def __init__(self):
        pass
    
    def fit(self, X):
        self.mean = np.mean(X, axis = 0)
        self.std = np.std(X, axis = 0)
        
    def transform(self, X):
       # print (self.mean)
       # print(X.shape[1])
       # dtype == int32 
        Y = np.arange(X.shape[0] * X.shape[1], dtype=float).reshape(X.shape[0],X.shape[1])
        assert self.mean.shape[0] == X.shape[1], "size not match"
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Y[i,j] = 1.0*(X[i, j] - self.mean[j])/self.std[j]
        return Y

if __name__ == "__main__" :
    x = np.arange(16).reshape(4,4)
    type(x)
    print(x.dtype)
    sc = scaling()
    sc.fit(x)
    x = sc.transform(x)        
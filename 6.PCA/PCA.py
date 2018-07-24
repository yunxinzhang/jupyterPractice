# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 15:20:42 2018

@author: zyx
"""

import numpy as np

class PCA():
    def __init__(self):
        pass
    
    def J(self, x,theta):
        #theta = theta/np.linalg.norm(theta)
        # norm == 1 is needed
        return x.dot(theta).dot(x.dot(theta))/len(x) 
    
    def DJ(self, x, theta):
        #theta = theta/np.linalg.norm(theta)
        # norm == 1 is needed
        return x.dot(theta).T.dot(x)/len(x)
    
    def demean(self, X):
        X -= np.mean(X,axis=0)
        return X
    
    def getTopOneDimension(self, X, niter = 1000, alpha = 0.01):
        cnt = 0
        theta = np.random.random(X.shape[1])
        theta = theta/np.linalg.norm(theta)
        while cnt < niter:
            dg = self.DJ(X, theta)
            theta += alpha*dg
            theta = theta/np.linalg.norm(theta)
            cnt += 1
        return theta
    # 降维过程中可能会 因为 没有 迭代到位 而计算所得为错误。 不好用
    def getTopNDimension(self, X1, n=1, niter=1000, alpha=0.01, sk = 10):
        res = []
        X = np.copy(X1) # 不快破坏原来的数据
        X = self.demean(X)
        theta = self.getTopOneDimension(X,niter,alpha)
        res.append(theta)
        cnt = 1
        while cnt < n:
            #一维的时候 numpy 自动将矩阵转化为 shape(n,), 无法进行矩阵运算。 需要reshape指明
            
            X -= (X.dot(theta).reshape(-1,1)).dot(theta.reshape(1,-1)).reshape(X.shape[0],-1)
            theta = self.getTopOneDimension(X,niter*sk,alpha)
            res.append(theta)
            cnt += 1
        return res
    # k 子空间维数
    # sk 子循环迭代增加倍数
    def fit(self, X, k=1,sk=10):
        res = self.getTopNDimension(X,k,sk)
        Mat = np.empty((k,X.shape[1]))
        for i in range(k):
            Mat[i] = res[i]
        self.Mat = Mat
        return self
    
    def transform(self, X):
        return X.dot(self.Mat.T)
    
    def getBack(self, X):
        return X.dot(self.Mat)
    
    
    
    
    
    
    
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 16:46:46 2018

@author: zyx
"""
import numpy as np

class GDCLF():
    
    def __init__(self):
        self.theta = None
        self.coef_ = None
        self.interceptor_ = None
    
    def getG(self ,x, y, theta):
        x2 = np.hstack((np.ones(len(x)).reshape(-1,1),x ))
        gd = np.zeros(len(theta))
        for i in range(x2.shape[1]):
            v = 0
            for j in range(len(x2)):
                v -= (y[j]-x2[j].dot(theta))*x2[j,i]
           # print(v,len(x))
            gd[i] = v/len(x2)
        #print(gd)   
        return gd
    # 向量化的方法来进行计算，效率提高十几倍
    def getGbyV(self, x, y, theta):
        x2 = np.hstack((np.ones(len(x)).reshape(-1,1),x ))
        #gd = np.zeros(len(theta))
        Y = [(y[i]-x2[i].dot(theta))*(-1) for i in range(len(x))]
        return x2.transpose().dot(Y)/len(x)
    # 循环性能差     1000 --->  10s
    def fit(self , x, y, alpha = 0.000001, niter = 100):
        self.theta = np.zeros(x.shape[1]+1)
        cnt = 0
        while cnt < niter:
            gd = self.getG(x, y, self.theta)
            self.theta -= alpha*gd
           # print(theta)
            cnt += 1
        self.interceptor_ = self.theta[0]
        self.coef_ = self.theta[1:]
        return self
    # 向量化的方法进行计算  1000 ---> 1s
    def fit2(self , x, y, alpha = 0.000001, niter = 100):
        self.theta = np.zeros(x.shape[1]+1)
        cnt = 0
        while cnt < niter:
            gd = self.getGbyV(x, y, self.theta)
            self.theta -= alpha*gd
           # print(theta)
            cnt += 1
        self.interceptor_ = self.theta[0]
        self.coef_ = self.theta[1:]
        return self
    # 为了随机梯度下降法
    def dj(self, xi, yi , theta):
        xi = np.hstack((np.array([1]).reshape(1,1),xi.reshape(1,-1)))
        return xi.T.dot(xi.dot(theta)-yi)
    # 随机梯度下降法  1000 ---> 20ms
    def fit3(self , x, y, alpha = 0.000001, niter = 100):
        self.theta = np.zeros(x.shape[1]+1)
        cnt = 0
        ind = np.arange(len(x))
        np.random.shuffle(ind)
        while cnt < niter:
            gd = self.dj(x[ind[cnt%len(x)],:], y[ind[cnt%len(x)]], self.theta)
            self.theta -= alpha*gd
           # print(theta)
            cnt += 1
        self.interceptor_ = self.theta[0]
        self.coef_ = self.theta[1:]
        return self
    def predict(self , x):
        return x.dot(self.coef_) + self.interceptor_
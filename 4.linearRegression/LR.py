# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 23:54:40 2018

@author: zyx
"""
import numpy as np

class SLR():
    def __init__(self):
        self.a_ = 0
        self.b_ = 0
    
    def fit(self, x, y):
        up, down = 0, 0
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        for i in range (len(y)):
            up += x[i]*y[i] - x[i]*y_mean
            down += x[i]*x[i] - x[i]*x_mean
        self.a_ = up/down
        self.b_ = -self.a_ * x_mean + y_mean
        return self
    
    def predict(self, x):
        return [self.a_ * xi + self.b_ for xi in x]
    
# 多重线性回归
class MLR():
    def __init__(self):
        pass
    #(x'x)^-1 *x'
    def fit(self, x, y):
        # np.hstack((a,b))
        x = np.hstack((np.ones(x.shape[0]).reshape(-1,1), x))
       # print(x)
       # print(x.transpose().dot(x))
       # ! 可能是奇异矩阵，无解。 而且相关的算法时间复杂度太高
        theta = np.linalg.inv(x.transpose().dot(x)).dot(x.transpose()).dot(y)
       # print(theta)
        self.interceptor_ = theta[0]
        self.coef_ = theta[1:]
        return self
    
    def predict(self, x):
        return [np.sum(self.coef_ * xi) + self.interceptor_ for xi in x]
    
if __name__ == "__main__":
    x = np.array([0,1,2,3,7,4]).reshape(3,2)
    y = np.array([1,2,5])
    mlr = MLR()
    mlr.fit(x,y)
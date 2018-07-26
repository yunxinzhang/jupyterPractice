# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 14:42:13 2018

@author: zyx
"""

import numpy as np

class LogisticRegressionMy():
    def __init__(self):
        self.theta = None
        self.coef_ = None
        self.interceptor_ = None

    def sigmoid(self,x):
        return 1.0/(1+np.exp(-x))
   
    def J(self, x, y, theta):
        sig = self.sigmoid(x.dot(theta))
        return -np.sum(y*(np.log(sig)) + (1-y)*(np.log(1-sig)))/len(x)
            
    def DJ(self, x, y, theta):
        return x.T.dot(self.sigmoid(x.dot(theta)) - y)/len(x)
    
    def gradientDescent(self, x, y, niter=1000, alpha=0.01):
        x_input = np.hstack((np.ones(len(x)).reshape(-1,1), x))
        self.theta = np.zeros(x.shape[1]+1)
        for i in range(niter):
            dg = self.DJ(x_input, y, self.theta)
            self.theta -= alpha * dg
        self.interceptor_ = self.theta[0]
        self.coef_ = self.theta[1:]
        return self
    def fit(self, x, y, niter=1000, alpha=0.01):
        x_input = np.hstack((np.ones(len(x)).reshape(-1,1), x))
        self.theta = np.zeros(x.shape[1]+1)
        for i in range(niter):
            dg = self.DJ(x_input, y, self.theta)
            self.theta -= alpha * dg
        self.interceptor_ = self.theta[0]
        self.coef_ = self.theta[1:]
        return self
    def predict(self,x):
        return x.dot(self.coef_) + self.interceptor_ >= 0 
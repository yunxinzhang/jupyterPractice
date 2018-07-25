# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 16:04:55 2018

@author: zyx
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def simplePolynomialRegression(x, y, n, plot=1):
    x_copy = np.copy(x).reshape(-1,1)
    x_input = np.copy(x_copy)
    for i in range(2, n+1):
        x_input = np.hstack((x_input, x_copy**i))
    xtr, xt, ytr, yt = train_test_split(x_input , y)
    sc = StandardScaler()
    sc.fit(xtr)
    xtr_standard = sc.transform(xtr)
    lr = LinearRegression()
    lr.fit(xtr_standard, ytr)
    xt_standard = sc.transform(xt)
    ac_train = lr.score(xtr_standard, ytr)
    ac_test  = lr.score(xt_standard, yt)
    print("train_ac", ac_train)
    print("test_ac: ", ac_test)
    if plot==1:
        yp = lr.predict(sc.transform(x_input))
        plt.scatter(x,y)
        plt.plot(x, yp, 'r')
    return ac_train, ac_test  
    
    
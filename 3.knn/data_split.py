# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 03:39:16 2018

@author: zyx
"""

import numpy as np

def data_split(X_input, Y_input, radio = 0.2, seed=None):
    
    assert X_input.shape[0] == Y_input.shape[0] , "shape must match"
    assert radio > 0 and radio < 1 , "radio must between 0 and 1"
    if seed :
        np.random.seed(seed)
    
    shuffled_index = np.random.permutation(X_input.shape[0])
    tsize = int(X_input.shape[0] * radio)
    
    X_train = X_input[shuffled_index[tsize:]]
    Y_train = Y_input[shuffled_index[tsize:]]
    
    X_test = X_input[shuffled_index[:tsize]]
    Y_test = Y_input[shuffled_index[:tsize]]

    return X_train, Y_train, X_test, Y_test    
    
def calc_ac(y_pred, y_real):
	cnt = sum([y_pred[i] == y_real[i] for i in range(len(y_pred))])
	return 1.0*cnt/len(y_pred)
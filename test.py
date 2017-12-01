# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 21:37:14 2017

@author: fuzhiqiang
"""
import numpy as np
from FML import loadmat
from FML import findknn
data = loadmat('Yale_32x32.mat')
a = data['fea']
row,col = a.shape
distlist = np.zeros([row,row])
b =[]
b=a[1,:]
#b = findknn(a)
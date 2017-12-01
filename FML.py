# 一个用来实现常用算法的库文件
import numpy as np


def loadmat(path):
    import scipy.io as sio
    data = sio.loadmat(path)
    return data


def findknn(data, k=2):
    row, col = data.shape
    distlist = np.zeros([row, row])
    numlist = np.zeros([row, k])
    for i in range(row):
        for j in range(row):
            distlist[i, j] = np.linalg.norm(data[i, :]-data[j, :])
        listnew = []
        listnew = distlist[i, :]
        listnew.sort()
        listnew = listnew[1:k+1]
        for l in range(k):
            local = np.where(distlist[i, :] == listnew[l])
            numlist[i, l] = local[0]
    return numlist

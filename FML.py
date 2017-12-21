# 一个用来实现常用算法的库文件
import numpy as np


def loadmat(path):
    import scipy.io as sio
    data = sio.loadmat(path)
    return data


def findknn(datain, k=2):
    row, col = datain.shape
    data = datain.astype(np.float64)
    distlist = np.zeros([row, row])
    numlist = np.zeros([row, k])
    for i in range(row):
        for j in range(row):
            distlist[i, j] = np.linalg.norm(data[i, :]-data[j, :])
        listnew = []
        listnew = np.sort(distlist[i, :])
        listnew = listnew[1:k+1]
        for l in range(k):
            local = np.where(distlist[i, :] == listnew[l])
            numlist[i, l] = local[0]
    return numlist


def constructW(datain):
    data = datain.astype(np.int32)
    row, col = data.shape
    w = np.zeros([row, row])
    for i in range(row):
        w[i, data[i, :]] = 1
    return w


def computeD(datain):
    result = datain.sum(axis=1) 
    result = np.diag(result)
    return result


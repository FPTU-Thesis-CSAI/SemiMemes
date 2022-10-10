
import numpy as np

def cal_single_instance(x, y):
    idx = np.argsort(-x) 
    y = y[idx]
    correct = 0
    prec = 0
    num = 0
    for i in range(x.shape[0]):
        if y[i] == 1:
            num += 1
            correct += 1
            prec += correct / (i + 1)
    return prec / num


def average_precision(x, y):
    """
    :param x: the predicted outputs of the classifier, the output of the ith instance for the jth class is stored in x(i,j)
    :param y: the actual labels of the instances, if the ith instance belong to the jth class, y(i,j)=1, otherwise y(i,j)=0
    :return: the average precision
    """
    n, d = x.shape
    if x.shape[0] != y.shape[0]:
        print("num of  instances for output and ground truth is different!!")
    if x.shape[1] != y.shape[1]:
        print("dim of  output and ground truth is different!!")
    aveprec = 0
    m = 0
    for i in range(n):
        s = np.sum(y[i])
        if s in range(1, d):
            aveprec += cal_single_instance(x[i], y[i])
            m += 1
    aveprec /= m
    return aveprec
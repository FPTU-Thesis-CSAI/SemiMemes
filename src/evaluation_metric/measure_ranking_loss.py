import numpy as np

def cal_single_instance(x, y):
    idx = np.argsort(x)  
    y = y[idx]
    m = 0
    n = 0
    rl = 0
    for i in range(x.shape[0]):
        if y[i] == 1:
            m += 1
        if y[i] == 0:
            rl += m
            n += 1
    rl /= (m * n)
    return rl


def ranking_loss(x, y):
    """
    :param x: the predicted outputs of the classifier, the output of the ith instance for the jth class is stored in x(i,j)
    :param y: the actual labels of the test instances, if the ith instance belong to the jth class, y(i,j)=1, otherwise x(i,j)=0
    :return: the ranking loss
    """
    n, d = x.shape
    if x.shape[0] != y.shape[0]:
        print("num of  instances for output and ground truth is different!!")
    if x.shape[1] != y.shape[1]:
        print("dim of  output and ground truth is different!!")
    m = 0
    rank_loss = 0
    for i in range(n):
        s = np.sum(y[i])
        if s in range(1, d):
            rank_loss += cal_single_instance(x[i], y[i])
            m += 1
    rank_loss /= m
    return rank_loss
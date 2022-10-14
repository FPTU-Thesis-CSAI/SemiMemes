import numpy as np

def cal_single_instance(x, y):
    idx = np.argsort(x)  # 升序排列
    y = y[idx]
    m = 0
    n = 0
    auc = 0
    for i in range(x.shape[0]):
        if y[i] == 1:
            m += 1
            auc += n
        if y[i] == 0:
            n += 1
    auc /= (m * n)
    return auc


def example_auc(x, y):
    """
    :param x: the predicted outputs of the classifier, the output of the ith instance for the jth class is stored in x(i,j)
    :param y: the actual labels of the instances, if the ith instance belong to the jth class, y(i,j)=1, otherwise y(i,j)=0
    :return: the example auc
    """
    n, d = x.shape
    if x.shape[0] != y.shape[0]:
        print("num of  instances for output and ground truth is different!!")
    if x.shape[1] != y.shape[1]:
        print("dim of  output and ground truth is different!!")
    m = 0
    auc = 0
    for i in range(n):
        s = np.sum(y[i])
        if s in range(1, d):
            auc += cal_single_instance(x[i], y[i])
            m += 1
    auc /= m
    return auc
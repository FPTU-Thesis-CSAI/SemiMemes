import numpy as np

def cal_single_label(x, y):
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


def macro_auc(x, y):
    """
    :param x: the predicted outputs of the classifier, the output of the ith instance for the jth class is stored in x(i,j)
    :param y: the actual labels of the instances, if the ith instance belong to the jth class, y(i,j)=1, otherwise y(i,j)=0
    :return: the macro auc
    """
    n, d = x.shape
    if x.shape[0] != y.shape[0]:
        print("num of  instances for output and ground truth is different!!")
    if x.shape[1] != y.shape[1]:
        print("dim of  output and ground truth is different!!")
    auc = 0
    num = 0
    for i in range(d):
        s = np.sum(y[:, i])
        if s in range(1, n):
            num += 1
            auc += cal_single_label(x[:, i], y[:, i])
    auc /= num
    return auc
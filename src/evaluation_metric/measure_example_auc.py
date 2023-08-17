import numpy as np
from torchmetrics.functional.classification import multilabel_auroc
import torch 

def auroc_score_pytorch(x,y, num_labels):
    # for i in range(x.shape[0]):
    #   for j in range(4):
    #     if x[i][j] >= 0.5:
    #       x[i][j] = 1
    #     else:
    #       x[i][j] = 0

    # multilabel_auroc(preds, target, num_labels=4, average="macro", thresholds=None)
    target = torch.FloatTensor(y)
    preds = torch.FloatTensor(x)
    return multilabel_auroc(preds, target, num_labels=num_labels, average="macro", thresholds=None)

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
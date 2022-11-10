import numpy as np 
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import torch 

def metrics(results, truths):
    preds = results
    truth = truths

    preds = np.where(preds > 0.5, 1, 0)
    truth = np.where(truth > 0.5, 1, 0)

    f_score_micro = f1_score(truth, preds, average='micro')
    f_score_macro = f1_score(truth, preds, average='macro')
    accuarcy = accuracy_score(truth, preds)
    recall = recall_score(truth, preds, average='weighted')
    precision = precision_score(truth, preds, average='weighted')
    return accuarcy, f_score_micro, f_score_macro, recall, precision

from torchmetrics import F1Score
from torchmetrics.classification import MultilabelF1Score
from sklearn.metrics import f1_score   

def f1_score_sklearn(x,y):
    x = x>0.5
    f1 = f1_score(y, x, average='macro')
    return f1

def f1_score_pytorch(x,y):
      metric = MultilabelF1Score(num_labels=4)
      target = torch.tensor(y)
      preds = torch.tensor(x)
      return metric(preds, target)

def macro_f1_multilabel(preds, target, num_labels=4, threshold = 0.5, reduce = True):
    preds = torch.tensor(preds)
    target = torch.tensor(target)

    preds = (preds > threshold).to(torch.long)
    target = target.to(torch.long)

    macro_f1 = F1Score(num_classes=2, average='macro')

    macro_f1_multilabel = []
    for i in range(num_labels):
        preds_label_i = preds[:, i]
        target_label_i = target[:, i]   
        macro_f1_score_i = macro_f1(preds_label_i, target_label_i)
        macro_f1_multilabel.append(macro_f1_score_i)

    if reduce:
        return torch.tensor(macro_f1_multilabel).mean().item()
    else:
        return torch.tensor(macro_f1_multilabel).numpy()
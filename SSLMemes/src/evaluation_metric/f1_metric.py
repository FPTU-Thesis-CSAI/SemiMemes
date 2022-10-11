import numpy as np 
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

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
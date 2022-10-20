from torchmetrics.classification import BinaryAUROC
import torch

metric = BinaryAUROC(thresholds=None)

def multilabel_binary_auroc(preds, target):
    score = dict()
    preds = torch.tensor(preds)
    target = torch.tensor(target)
    for label in range(target.shape[1]):
        pred_label = preds[:, label]
        target_label = target[:, label]
        score[label] = metric(pred_label, target_label)

    return score




# target = torch.tensor([[1, 1, 0, 0],
#                         [1, 1, 0, 0],
#                         [1, 0, 0, 0],
#                         [0, 1, 0, 0],
#                         [1, 1, 0, 0],
#                         [1, 0, 0, 0]])

# preds = torch.tensor([[0.5222886 , 0.49643162, 0.47821411, 0.5259715 ],
#                         [0.5167473 , 0.48927295, 0.48414344, 0.52201885],
#                         [0.5191    , 0.49132732, 0.48287907, 0.5257748 ],
#                         [0.52028465, 0.49342018, 0.48145986, 0.5263021 ],
#                         [0.5197327 , 0.4926404 , 0.48086417, 0.5253642 ],
#                         [0.5186344 , 0.49012083, 0.4831387 , 0.52360713]])

# print(multilabel_binary_auroc(preds, target))

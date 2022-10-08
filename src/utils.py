from torchmetrics import AUROC

def auroc(preds,target):
    auroc = AUROC()
    score_info = []
    t_score = 0
    num_label = target.size()[1]
    for i in range(num_label):
        score = auroc(preds[:,i], target[:,i])
        t_score += score
        score_info.append(score)
    score_info.append(t_score/num_label)
    print(f"score for class 1:{score_info[0]}, class 2:{score_info[1]}, class 3:{score_info[2]}, class 4:{score_info[3]},avg score:{score_info[4]}")
    return score_info[4]
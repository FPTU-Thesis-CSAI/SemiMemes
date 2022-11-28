import mmcv 
import numpy as np  
import pandas as pd 

def dump_freq_data(meta_data_path, label_cols):
    metadata = pd.read_csv(meta_data_path)
    labels = metadata[label_cols].values
    num_classes = 4
    co_labels = [[] for _ in range(num_classes)]
    condition_prob = np.zeros([num_classes, num_classes])
    for label in labels:
        for idx in np.where(np.asarray(label) == 1)[0]:
            co_labels[idx].append(label)

    for cla in range(num_classes):
        cls_labels = co_labels[cla]
        num = len(cls_labels)
        condition_prob[cla] = np.sum(np.asarray(cls_labels), axis=0) / num
    
    class_freq = np.sum(labels, axis=0)
    neg_class_freq = np.shape(labels)[0] - class_freq
    path = 'data/class_freq.pkl'
    save_data = dict(gt_labels=labels, class_freq=class_freq, neg_class_freq=neg_class_freq
                        , condition_prob=condition_prob)
    mmcv.dump(save_data, path)
    return path
    
if __name__ == '__main__':
    meta_data_path = 'data/MAMI_processed/train_labeled_ratio-0.05.csv'
    label_cols = ["shaming","stereotype","objectification","violence"]
    dump_freq_data(meta_data_path, label_cols)
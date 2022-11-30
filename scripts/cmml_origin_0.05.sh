CUDA_VISIBLE_DEVICES=$1 python src/CmmlTrain.py \
    --experiment cmml_origin_0.05 \
    --train_labeled_csv 'data/MAMI_processed/train_labeled_ratio-0.05.csv' \
    --train_unlabeled_csv 'data/MAMI_processed/train_unlabeled_ratio-0.05.csv' \
    --use-deep-weak-attention \
    --use-step-lr \
    --batchsize 40  \
    --epochs 50 \
    --lr-supervise 0.0001 \
    --use-bce-loss \
    --wd 1e-4 \
    --use-adam \
    --Predictpara '256, 4' \
    --use-sentence-vectorizer True \
    --resnet-model resnet18 \


CUDA_VISIBLE_DEVICES=$1 python src/train_2_stages.py \
    --experiment 0.3_ours_bce \
    --config_yaml_path 'config/0.3label.yaml' \
    --use-step-lr \
    --batchsize 40  \
    --epochs 50 \
    --lr-supervise 0.0001 \
    --use-bce-loss \
    --use-adam \
    --pretrain_auto_encoder True \


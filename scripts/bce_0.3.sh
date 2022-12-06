CUDA_VISIBLE_DEVICES=$1 python src/train_model_concat.py \
    --experiment 0.3_bce \
    --config_yaml_path 'config/0.3label_bce_ae.yaml' \
    --use-step-lr \
    --batchsize 40  \
    --epochs 50 \
    --lr-supervise 0.0001 \
    --use-bce-loss \
    --use-adam \
    # --pretrain_auto_encoder True \


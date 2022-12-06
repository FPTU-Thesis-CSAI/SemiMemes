CUDA_VISIBLE_DEVICES=$1 python src/train_model_concat.py \
    --experiment 0.1_bce \
    --config_yaml_path 'config/0.1label.yaml' \
    --use-step-lr \
    --batchsize 40  \
    --lr-supervise 0.0001 \
    --use-bce-loss \
    --use-adam \
    --pretrain_auto_encoder True \


CUDA_VISIBLE_DEVICES=$1 python src/train_2_stages.py \
    --experiment 0.05_ours \
    --config_yaml_path 'config/0.05label.yaml' \
    --use-step-lr \
    --batchsize 40  \
    --lr-supervise 0.0001 \
    --use-resample-loss \
    --use-adam \
    --pretrain_auto_encoder True \


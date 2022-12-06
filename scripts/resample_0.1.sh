CUDA_VISIBLE_DEVICES=$1 python src/train_model_concat.py \
    --experiment 0.1_resample \
    --config_yaml_path 'config/0.1label.yaml' \
    --use-step-lr \
    --batchsize 40  \
    --epochs 50 \
    --lr-supervise 0.0001 \
    --use-resample-loss \
    --use-adam \
    # --pretrain_auto_encoder True \


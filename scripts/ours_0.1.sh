CUDA_VISIBLE_DEVICES=$1 python src/train_2_stages.py \
    --experiment 0.1_ours_highgamma_norm_feature \
    --config_yaml_path 'config/0.1label.yaml' \
    --use-step-lr \
    --batchsize 40  \
    --epochs 50 \
    --lr-supervise 0.0001 \
    --use-resample-loss \
    --use-adam \
    --pretrain_auto_encoder True \


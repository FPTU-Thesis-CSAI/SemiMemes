
# CUDA_VISIBLE_DEVICES=$1 python src/CmmlTrain.py \
#     --experiment concat_modalities \
#     --use-augmentation \
#     --use-multi-step-lr \
#     --lr-supervise 1e-4  \
#     --use-drop-out \
#     --not-calculate-txt-img-loss \
#     --use-clip-norm  \
#     --use-bce-loss \
#     --use-act \
#     --wd 1e-4 \
#     --use-concat-modalities \
#     --use-adam \
#     --Predictpara '512, 5' \
#     --modality-project-dim 5  \
#     --use-one-head \

# CUDA_VISIBLE_DEVICES=$1 python src/CmmlTrain.py \
#     --experiment resample_loss_strike_2 \
#     --use-augmentation \
#     --use-sgd \
#     --use-linear-scheduler \
#     --lr-supervise 0.02  \
#     --use-drop-out  \
#     --use-resample-loss \
#     --use-coattention \
#     --use-act  \
#     --wd 1e-4 \
#     --Predictpara '256, 5' \
#     --epochs 150 

CUDA_VISIBLE_DEVICES=$1 python src/CmmlTrain.py \
    --experiment resample_loss_strike_neg_scale_3 \
    --use-augmentation \
    --use-adam \
    --use-step-lr \
    --lr-supervise 0.0001  \
    --use-drop-out \
    --use-resample-loss \
    --use-coattention \
    --use-act  \
    --wd 1e-4 \
    --Predictpara '256, 5' \
    --Textpredictpara '256, 5' \
    --Imgpredictpara '256, 5' \
    --epochs 150

# CUDA_VISIBLE_DEVICES=$1 python src/CmmlTrain.py \
#     --experiment resample_loss_strike_neg_scale_3 \
#     --use-augmentation \
#     --use-sgd \
#     --use-linear-scheduler \
#     --lr-supervise 0.02  \
#     --use-drop-out \
#     --use-resample-loss \
#     --use-coattention \
#     --use-act  \
#     --wd 1e-4 \
#     --Predictpara '256, 5' \
#     --neg-scale 7.0 \
#     --epochs 150

#total weight:0.7183905025643998, img weight:0.2403833052800633, text weight:0.041226192155536835 (clip,sgd,deep_weak_attention,resampleloss) 
#total weight:0.7966240742351592, img weight:0.10191846424448874, text weight:0.10145746152035197 (concat,clip model, adam,resampleloss)
#(clip,adam,deep_weak_attention,resampleloss,mORG)
#(clip,adam,deep_weak_attention,resampleloss,oORG)

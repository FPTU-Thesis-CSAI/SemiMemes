
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
#total weight:0.0748026304586017, img weight:0.6779435845145609, text weight:0.24725378502683737 (clip,adam,deep_weak_attention,resampleloss,mORG)
#total weight:0.048916584058129445, img weight:0.9453698511056364, text weight:0.005713564836234145 (clip,adam,deep_weak_attention,resampleloss,oORG)
#total weight:0.06643208080547996, img weight:0.5336646104507087, text weight:0.39990330874381136
#total weight:0.6194652844102732, img weight:0.16314931594158397, text weight:0.2173853996481428
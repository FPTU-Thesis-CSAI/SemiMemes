import torch
from torch import nn

def deep_weak_attention(attention_model, modalities, cuda):
    modality_attention = []
    for modal in modalities:
        modal_k = attention_model(modal)
        modality_attention.append(modal_k)
    modality_attention = torch.cat(modality_attention, 1)
    modality_attention = nn.functional.softmax(modality_attention, dim = 1)
    
    modal_attention_list = []
    for i, modal in enumerate(modalities):
        modal_attention = torch.zeros(1, len(modal))
        modal_attention[0] = modality_attention[:,i]
        modal_attention = modal_attention.t()
        if cuda:
            modal_attention = modal_attention.cuda()
        
        modal_attention_list.append(modal_attention)
    
    supervise_feature_hidden = torch.stack([modal * modal_attention_list[i] for i, modal in enumerate(modalities)], axis=0).sum(axis=0)
    
    return supervise_feature_hidden, modality_attention
    
    # img_attention = torch.zeros(1, len(y))
    # img_attention[0] = modality_attention[:,0]
    # img_attention = img_attention.t()
    # text_attention = torch.zeros(1, len(y))
    # text_attention[0] = modality_attention[:,1]
    # text_attention = text_attention.t()
    
#     if args.use_caption:
#         caption_attention = torch.zeros(1, len(y))
#         caption_attention[0] = modality_attention[:,2]
#         caption_attention = caption_attention.t()
#         if cuda:
#             caption_attention = caption_attention.cuda()
#         supervise_feature_hidden = img_attention * supervise_imghidden + text_attention * supervise_texthidden + caption_attention * supervise_caption_hidden
#     else:
#         supervise_feature_hidden = img_attention * supervise_imghidden + text_attention * supervise_texthidden
                    
#     supervise_predict = model.Predictmodel(supervise_feature_hidden)

    # if cuda:
    #     img_attention = img_attention.cuda()
    #     text_attention = text_attention.cuda()
    # supervise_feature_hidden = img_attention * supervise_imghidden + text_attention * supervise_texthidden
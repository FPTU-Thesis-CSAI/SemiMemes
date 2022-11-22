from arguments import get_args 
import os  
import torch 
from data.dataClass import MemotionDatasetForCmml
from model.CmmlLayer import CmmlModel
# from model.eman import EMAN 
import torch.optim as optim 
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from torch.utils.data import DataLoader 
import datetime 
import numpy as np  
from test import test_multilabel
from tqdm import tqdm
from loss import focal_binary_cross_entropy, diversity_measurement, consistency_measurement
from loss import focal_binary_cross_entropy,zlpr_loss,AsymmetricLoss,ResampleLoss 
import torch.nn as nn
import wandb
import random
from matplotlib import pyplot as plt
import torch.nn as nn
from utils.plot_utils import get_confusion_matrix
from torch.cuda.amp import autocast, GradScaler
import clip
import open_clip
from model.clip_info import clip_nms,clip_dim

from model.utils import LARS,adjust_learning_rate,exclude_bias_and_norm
from ORG import GB_estimate 
from copy import deepcopy 
import gc 
from data.semi_supervised_data import *
from utils.npy_save import npy_save_txt

from model.dualstream_net import CmmlModel_v2
from model.deep_weak_attention import deep_weak_attention


def train(args,model, dataset,
        supervise_epochs = 200, text_supervise_epochs = 50, img_supervise_epochs = 50, 
        lr_supervise = 0.01, text_lr_supervise = 0.0001, img_lr_supervise = 0.0001,
        weight_decay = 0, batchsize = 32,lambda1=0.01,lambda2=1, textbatchsize = 32,
        imgbatchsize = 32, cuda = False, savepath = '',eman=None): 
    
    model.train()
    print("train")
    loss = 0
    cita = 1.003
    if args.use_lars_optimizer:
        print("==============use lars optimizer===============")
        optimizer = LARS(
            model.parameters(),
            lr=0,
            weight_decay=args.weight_decay,
            weight_decay_filter=exclude_bias_and_norm,
            lars_adaptation_filter=exclude_bias_and_norm,
        )
    elif args.use_sgd:
        print("==============use sgd optimizer===============")
        optimizer = torch.optim.SGD(model.parameters(), lr=lr_supervise, weight_decay=weight_decay)
    elif args.use_adam:
        print("==============use adam optimizer===============")
        optimizer = optim.Adam(model.parameters(), lr = lr_supervise, weight_decay = weight_decay)
    
    if args.use_linear_scheduler:
        print("==============use linear lr scheduler===============")
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,start_factor=1./3,total_iters=80)
    elif args.use_step_lr:
        print("==============use step scheduler===============")
        scheduler = StepLR(optimizer, step_size = 100, gamma = 0.9)  
    elif args.use_multi_step_lr:
        print("==============use multi step scheduler===============")
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [5,10,15],gamma=0.5)

    if args.use_bce_loss:
        print("==============use bce loss===============")
        criterion = torch.nn.BCELoss()
    elif args.use_focal_loss:
        print("==============use focal loss===============")
        criterion = partial(focal_binary_cross_entropy,args)
    elif args.use_zlpr_loss:
        print("==============use zlpr loss===============")
        criterion = zlpr_loss
    elif args.use_asymmetric_loss:
        print("==============use asymmetric loss===============")
        criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
    elif args.use_resample_loss:
        print("==============use resample loss===============")
        criterion = ResampleLoss(args)
    sigmoid = torch.nn.Sigmoid()
    if args.use_org:
        print("==============use ORG===============")
        estimate_model = deepcopy(model)
        modality_weights = GB_estimate(args,estimate_model,supervise_epochs,dataset,optimizer,scheduler,criterion,cuda,sigmoid)
        del estimate_model
        gc.collect()
        print(f"total weight:{modality_weights[0]}, img weight:{modality_weights[1]}, text weight:{modality_weights[2]}")
    for epoch in range(1, supervise_epochs + 1):
        epoch_supervise_loss_train = 0
        epoch_div_train = 0 
        epoch_unsupervise_loss_train = 0
        epoch_v_supervise_loss_train = 0 
        epoch_c_supervise_loss_train = 0
        epoch_v_unsupervise_loss_train = 0
        epoch_c_unsupervise_loss_train = 0
        epoch_img_loss_train = 0
        epoch_text_loss_train = 0
        epoch_total_loss_train = 0
        if args.use_sim_loss:
            epoch_i_supervise_loss_train = 0
            epoch_i_unsupervise_loss_train = 0            
        num_supervise_sample = 0
        num_unsupervise_sample = 0

        num_steps = min(len(dataset['train_sup']), len(dataset['train_unsup']))

        for step, (supbatch, unsupbatch) in tqdm(enumerate(zip(dataset['train_sup'], dataset['train_unsup']), start=1),desc=f'epoch {epoch}',
                                                total=min(len(dataset['train_sup']), len(dataset['train_unsup']))):
            # print(batch_index)

            (sup_img, sup_text), sup_label = supbatch
            (unsup_img, unsup_text) = unsupbatch

            # for single label
            # sup_label = sup_label.unsqueeze(-1) if len(sup_label.shape) == 1 else sup_label # expand last dim for single label only
            # sup_label = torch.stack([1-sup_label, sup_label], axis=-1)
            # for single label

            scheduler.step()
            y = sup_label
            '''
            Attention architecture and use bceloss.
            '''
            supervise_img_xx = sup_img
            if args.use_sentence_vectorizer:
                # print("============use sentence bert vectorizer===============")
                supervise_text_xx = sup_text['sentence_vectors'].float()
                supervise_text_xx = Variable(supervise_text_xx).cuda() if cuda else Variable(supervise_text_xx)  
            if args.use_bert_embedding:
                # print("===============use bert embedding================")
                supervise_bert_xx = sup_text['sbert_embedding']
            label = sup_label

            if args.use_bert_embedding:
                supervise_bert_xx = sup_text['sbert_embedding']
                supervise_bert_xx = Variable(supervise_bert_xx).cuda() if cuda else Variable(supervise_bert_xx)
            
            if args.use_bert_model:
                # print("===============use bert model================")
                supervise_input_ids = sup_text['input_ids']
                supervise_attn_mask = sup_text['attention_mask']
                supervise_input_ids = supervise_input_ids.long()
                supervise_attn_mask = supervise_attn_mask.long()
                supervise_input_ids = Variable(supervise_input_ids).cuda() if cuda else Variable(supervise_input_ids)
                supervise_attn_mask = Variable(supervise_attn_mask).cuda() if cuda else Variable(supervise_attn_mask)

            if args.use_clip:
                # print("===============use clip token================")
                supervise_clip_ids = sup_text['clip_tokens']
                supervise_clip_ids = Variable(supervise_clip_ids).cuda() if cuda else Variable(supervise_clip_ids)   
            
            if args.use_caption:
                supervise_clip_ids_caption = sup_text['caption_clip_tokens']
                supervise_clip_ids_caption = Variable(supervise_clip_ids_caption).cuda() if cuda else Variable(supervise_clip_ids_caption)
                supervise_caption_hidden = model.Captionfeaturemodel(clip_input_ids=supervise_clip_ids_caption)
                supervise_captionpredict = model.Captionpredictmodel(supervise_caption_hidden)
                captionloss = criterion(supervise_captionpredict, label)

            supervise_img_xx = supervise_img_xx.float()
            label = label.float()
            supervise_img_xx = Variable(supervise_img_xx).cuda() if cuda else Variable(supervise_img_xx)  
                
            label = Variable(label).cuda() if cuda else Variable(label)  
            
            # #=========== DEFAULT FORWARD ===========#
            supervise_imghidden = model.Imgmodel(supervise_img_xx)
            if args.use_bert_embedding:
                # supervise_texthidden = model.Textfeaturemodel(x = supervise_text_xx,bert_emb = supervise_bert_xx)
                supervise_texthidden = model.Textfeaturemodel(x = supervise_bert_xx)
            if args.use_clip:
                supervise_texthidden = model.Textfeaturemodel(clip_input_ids = supervise_clip_ids)
            elif args.use_bert_embedding:
                supervise_texthidden = model.Textfeaturemodel(x = supervise_text_xx,bert_emb = supervise_bert_xx)
            elif args.use_bert_model:
                supervise_texthidden = model.Textfeaturemodel(input_ids = supervise_input_ids,attn_mask = supervise_attn_mask)
            else:
                supervise_texthidden = model.Textfeaturemodel(x = supervise_text_xx)
            
            if args.multi_scale_fe:
                supervise_imghidden, attn_w = model.ImgAttention(query=supervise_texthidden.unsqueeze(1), 
                                                                key=supervise_imghidden,
                                                                value=supervise_imghidden)
                supervise_imghidden = supervise_imghidden.squeeze(1)
            # #=========== DEFAULT FORWARD ===========#
            
            supervise_imgpredict = model.Imgpredictmodel(supervise_imghidden)
            supervise_textpredict = model.Textpredictmodel(supervise_texthidden)
            
            modalities = [supervise_imghidden, supervise_texthidden]
            if args.use_caption:
                modalities.append(supervise_caption_hidden)
            
            if args.use_deep_weak_attention:
                # print("===============use deep weak attention================")
                supervise_feature_hidden, modal_att_w = deep_weak_attention(model.Attentionmodel, modalities, cuda)
                
            elif args.use_coattention:
                # print("===============use co-attention================")
                supervise_feature_hidden = model.FusionCoattention(supervise_imghidden,supervise_texthidden)
            elif args.use_concat_modalities:
                # print("===============use concat================")
                supervise_feature_hidden = torch.cat(tuple(modalities), dim=1)
                
            if args.use_vicreg_in_training:
                # print("===============use vcreg================")
                vcreg_loss_supervise_img_text = model.ProjectormodelImgText(supervise_imghidden,supervise_texthidden)
                vcreg_loss_supervise_img_total = model.ProjectormodelImgTotal(supervise_imghidden,supervise_feature_hidden)
                vcreg_loss_supervise_text_total = model.ProjectormodelTextTotal(supervise_feature_hidden,supervise_texthidden)

            supervise_predict = model.Predictmodel(supervise_feature_hidden)       
            if args.use_focal_loss or  args.use_bce_loss:
                supervise_predict = sigmoid(supervise_predict)
                totalloss = criterion(supervise_predict, label)
                supervise_textpredict = sigmoid(supervise_textpredict)
                supervise_imgpredict = sigmoid(supervise_imgpredict)                
                imgloss = criterion(supervise_imgpredict, label)
                textloss = criterion(supervise_textpredict, label)
            elif args.use_zlpr_loss or args.use_asymmetric_loss or args.use_resample_loss:
                imgloss = criterion(supervise_imgpredict, label)
                textloss = criterion(supervise_textpredict, label)
                totalloss = criterion(supervise_predict, label)
            '''
            Diversity Measure code.
            '''        
            div = nn.CosineSimilarity(dim=1)(supervise_imgpredict, supervise_textpredict).mean(axis=0)

            '''
            Diversity Measure code.
            ''' 

            
            # if args.use_caption:
            #     supervise_loss = imgloss + textloss + 3.0*totalloss + captionloss
            # else:
            #     supervise_loss = imgloss + textloss + 2.0*totalloss

            if args.use_org:
                supervise_loss = modality_weights[1]*imgloss + modality_weights[2]*textloss + modality_weights[0]*totalloss
            else:
                supervise_loss = imgloss + textloss + 2.0*totalloss
                
                if args.use_caption:
                    supervise_loss = imgloss + textloss + 3.0*totalloss
                
            if args.use_vicreg_in_training:
                supervise_loss += sum(vcreg_loss_supervise_img_text)+sum(vcreg_loss_supervise_img_total)+sum(vcreg_loss_supervise_text_total)

            epoch_img_loss_train += imgloss.item()
            epoch_text_loss_train += textloss.item() 
            epoch_total_loss_train += totalloss.item()
            #=======================================#

            # ================== UNSUPERVISE =================== # 

            unsupervise_img_xx = unsup_img
            if args.use_sentence_vectorizer:
                unsupervise_text_xx = unsup_text['sentence_vectors'].float()
                unsupervise_text_xx = Variable(unsupervise_text_xx).cuda() if cuda else Variable(unsupervise_text_xx) 

            if args.use_bert_embedding:
                unsupervise_bert_xx = unsup_text['sbert_embedding']
                unsupervise_bert_xx = Variable(unsupervise_bert_xx).cuda() if cuda else Variable(unsupervise_bert_xx)
            
            if args.use_bert_model:
                unsupervise_token_xx = unsup_text['input_ids']
                unsupervise_attn_mask_xx = unsup_text['attention_mask']
                unsupervise_token_xx = unsupervise_token_xx.long()
                unsupervise_attn_mask_xx = unsupervise_attn_mask_xx.long()
                unsupervise_token_xx = Variable(unsupervise_token_xx).cuda() if cuda else Variable(unsupervise_token_xx) 
                unsupervise_attn_mask_xx = Variable(unsupervise_attn_mask_xx).cuda() if cuda else Variable(unsupervise_attn_mask_xx) 
                
            if args.use_clip:
                unsupervise_clip_ids = unsup_text['clip_tokens']
                unsupervise_clip_ids = Variable(unsupervise_clip_ids).cuda() if cuda else Variable(unsupervise_clip_ids)    

            unsupervise_img_xx = unsupervise_img_xx.float()
            # unsupervise_text_xx = unsupervise_text_xx.float()
            unsupervise_img_xx = Variable(unsupervise_img_xx).cuda() if cuda else Variable(unsupervise_img_xx)     
            # unsupervise_text_xx = Variable(unsupervise_text_xx).cuda() if cuda else Variable(unsupervise_text_xx) 

            # #=========== DEFAULT FORWARD ===========#
            unsupervise_imghidden = model.Imgmodel(unsupervise_img_xx)
            if args.use_bert_embedding:
                # unsupervise_texthidden = model.Textfeaturemodel(x = unsupervise_text_xx,bert_emb = unsupervise_bert_xx)
                unsupervise_texthidden = model.Textfeaturemodel(x = unsupervise_bert_xx)
            if args.use_clip:
                unsupervise_texthidden = model.Textfeaturemodel(clip_input_ids = unsupervise_clip_ids)
            elif args.use_bert_embedding:
                unsupervise_texthidden = model.Textfeaturemodel(x = unsupervise_text_xx,bert_emb = unsupervise_bert_xx)
            elif args.use_bert_model:
                unsupervise_texthidden = model.Textfeaturemodel(input_ids = unsupervise_token_xx,bert_emb = unsupervise_attn_mask_xx)
            else:
                unsupervise_texthidden = model.Textfeaturemodel(x = unsupervise_text_xx)
                # unsupervise_texthidden = model.Textfeaturemodel(x = unsupervise_bert_xx)

            if args.multi_scale_fe:
                unsupervise_imghidden, attn_w = model.ImgAttention(query=unsupervise_texthidden.unsqueeze(1), 
                                                                    key=unsupervise_imghidden,
                                                                    value=unsupervise_imghidden)
                unsupervise_imghidden = unsupervise_imghidden.squeeze(1)
            # #=========== DEFAULT FORWARD ===========#

            if args.use_vicreg_in_training:
                # print("===============use vcreg in unsupervised data================")
                vcreg_loss_unsupervise_img_text = model.ProjectormodelImgText(unsupervise_imghidden,unsupervise_texthidden)

            unsupervise_imgpredict = model.Imgpredictmodel(unsupervise_imghidden)
            unsupervise_textpredict = model.Textpredictmodel(unsupervise_texthidden)

            '''
            Robust Consistency Measure code.
            '''
            unsupervise_loss = consistency_measurement(unsupervise_imgpredict, unsupervise_textpredict, cita=cita)
            '''
            Robust Consistency Measure code.
            '''

            total_loss = supervise_loss + 0.01* div +  unsupervise_loss
            if args.use_vicreg_in_training:
                total_loss += sum(vcreg_loss_unsupervise_img_text)
            
            epoch_supervise_loss_train += supervise_loss.item()
            epoch_div_train += div.item() 
            epoch_unsupervise_loss_train += unsupervise_loss.item()

            # ================== UNSUPERVISE =================== #
            
            loss += total_loss.item()
            optimizer.zero_grad()
            total_loss.backward()
            if args.use_clip_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # if args.use_multi_step_lr or args.use_linear_scheduler:
        #     scheduler.step()

        #===================== Multilabel =================#
        (f1_macro_multi_total, f1_macro_multi_img, f1_macro_multi_text,
        f1_weighted_multi_total,f1_weighted_multi_img,f1_weighted_multi_text,
        auc_pm1,auc_pm2,auc_pm3,
        total_predict, truth,humour,sarcasm,offensive,motivational,humour_truth,
        sarcasm_truth,offensive_truth,motivational_truth) = test_multilabel(args,model, dataset['val'], batchsize = batchsize, cuda = cuda)

        total_step = num_steps
        epoch_supervise_loss_train = epoch_supervise_loss_train/total_step
        epoch_div_train = epoch_div_train/total_step
        epoch_unsupervise_loss_train = epoch_unsupervise_loss_train/total_step
        epoch_img_loss_train = epoch_img_loss_train/total_step
        epoch_text_loss_train = epoch_text_loss_train/total_step 
        epoch_total_loss_train = epoch_total_loss_train/total_step


        print("epoch_total_loss_train:",epoch_total_loss_train)
        print("epoch_img_loss_train",epoch_img_loss_train,
            "\t epoch_text_loss_train:",epoch_text_loss_train,
            "\t epoch_total_loss_train:",epoch_total_loss_train)
        
        print("epoch_supervise_loss_train:",epoch_supervise_loss_train,
        '\t epoch_div_train:',epoch_div_train,'\t epoch_unsupervise_loss_train',epoch_unsupervise_loss_train)
                
        print("f1_macro_multi_1:",f1_macro_multi_total,
        "\t f1_macro_multi_2",f1_macro_multi_img,
        "\t f1_macro_multi_3:",f1_macro_multi_text)

        wandb.log({"learning rate/lr":scheduler.get_last_lr()[0]})
        wandb.log({"f1_macro_multi_total":f1_macro_multi_total})
        wandb.log({"f1_macro_multi_img":f1_macro_multi_img})
        wandb.log({"f1_macro_multi_text":f1_macro_multi_text})

        wandb.log({"f1_weighted_multi_total":f1_weighted_multi_total})
        wandb.log({"f1_weighted_multi_img":f1_weighted_multi_img})
        wandb.log({"f1_weighted_multi_text":f1_weighted_multi_text})
        
        wandb.log({"f1_weighted_multi_total":f1_weighted_multi_total})
        wandb.log({"f1_weighted_multi_img":f1_weighted_multi_img})
        wandb.log({"f1_weighted_multi_text":f1_weighted_multi_text})
        
        print(f"[F1 Macro multilabel] Total: {f1_macro_multi_total} Image {f1_macro_multi_img} Text {f1_macro_multi_text}")
        print(f"[F1 weight multilabel] Total: {f1_weighted_multi_total} Image {f1_weighted_multi_img} Text {f1_weighted_multi_text}")


        total_2 = (total_predict > 0.5).astype('int')

        get_confusion_matrix(truth[:,0],total_2[:,0])
        wandb.log({"confusion_matrix_shaming": plt})
        get_confusion_matrix(truth[:,1],total_2[:,1])
        wandb.log({"confusion_matrix_stereotype": plt})
        get_confusion_matrix(truth[:,2],total_2[:,2])
        wandb.log({"confusion_matrix_objectification": plt})
        get_confusion_matrix(truth[:,3],total_2[:,3])
        wandb.log({"confusion_matrix_violence": plt})

        wandb.log({"histogram/_hist_label_shaming_pred":wandb.Histogram(np_histogram = humour)})
        wandb.log({"histogram/_hist_label_stereotype_pred":wandb.Histogram(np_histogram = sarcasm)})
        wandb.log({"histogram/_hist_label_objectification_pred":wandb.Histogram(np_histogram = offensive)})
        wandb.log({"histogram/_hist_label_violence_pred":wandb.Histogram(np_histogram = motivational)})

        wandb.log({"histogram/_hist_label_shaming_truth":wandb.Histogram(np_histogram = humour_truth)})
        wandb.log({"histogram/_hist_label_stereotype_truth":wandb.Histogram(np_histogram = sarcasm_truth)})
        wandb.log({"histogram/_hist_label_objectification_truth":wandb.Histogram(np_histogram = offensive_truth)})
        wandb.log({"histogram/_hist_label_violence_truth":wandb.Histogram(np_histogram = motivational_truth)})

        print('rocauc_pm:    ', auc_pm1,'\t', auc_pm2,'\t', auc_pm3)

        # if not args.use_one_head: 
        (f1_macro_multi_total, f1_macro_multi_img, f1_macro_multi_text,
        f1_weighted_multi_total,f1_weighted_multi_img,f1_weighted_multi_text,
        auc_pm1,auc_pm2,auc_pm3,
        total_predict, truth,humour,sarcasm,offensive,motivational,humour_truth,
        sarcasm_truth,offensive_truth,motivational_truth) = test_multilabel(args,model, dataset['test'], batchsize = batchsize, cuda = cuda)
        # else:
        #     (f1_macro_multi_total,f1_weighted_multi_total,auc_pm1,
        #     total_predict, truth, humour,sarcasm,offensive,motivational,humour_truth,
        #     sarcasm_truth,offensive_truth,motivational_truth) = test_multilabel(args,model.Textfeaturemodel,
        #     None,None, model.Imgmodel,
        #     model.Predictmodel, model.Attentionmodel, dataset['test'], batchsize = batchsize, cuda = cuda)

        wandb.log({"f1_macro_multi_total_test":f1_macro_multi_total})
        # if not args.use_one_head:
        wandb.log({"f1_macro_multi_img_test":f1_macro_multi_img})
        wandb.log({"f1_macro_multi_text_test":f1_macro_multi_text})

        wandb.log({"f1_weighted_multi_total_test":f1_weighted_multi_total})
        # if not args.use_one_head:
        wandb.log({"f1_weighted_multi_img_test":f1_weighted_multi_img})
        wandb.log({"f1_weighted_multi_text_test":f1_weighted_multi_text})
        
        # if args.use_one_head:
        #     print(f"Test [F1 Macro multilabel] Total: {f1_macro_multi_total}")
        #     print(f"[F1 weight multilabel] Total: {f1_weighted_multi_total}")
        # else:
        #     print(f"Test [F1 Macro multilabel] Total: {f1_macro_multi_total} Image {f1_macro_multi_img} Text {f1_macro_multi_text}")
        #     print(f"[F1 weight multilabel] Total: {f1_weighted_multi_total} Image {f1_weighted_multi_img} Text {f1_weighted_multi_text}")

    return 


if __name__ == '__main__':
    seed = 42
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Torch RNG
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Python RNG
    np.random.seed(seed)
    random.seed(seed)

    args = get_args()

    wandb.login(key = 'd87822d5fa951a22676b0985f891c9021b875ae3')
    # wandb.init(project="meme_experiments", entity="meme-analysts", mode="disabled")
    wandb.init(project="meme_experiments", entity="meme-analysts")
    # wandb.init()

    wandb.run.name = args.experiment
    print(f"Experiement: {wandb.run.name}")

    if args.use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpu
        cuda = torch.cuda.is_available() and args.use_gpu
    else:
        cuda = False

    input_resolution = None
    clip_model = None
    cdim = None
    if args.use_clip:
        print("==============use clip model===============")
        if args.use_open_clip:
            clip_model, _, _ = open_clip.create_model_and_transforms(args.clip_model,pretrained=args.clip_pretrained)
            input_resolution = clip_model.visual.image_size[0]
        else:
            clip_model, _ = clip.load(clip_nms[args.clip_model],jit=False)
            clip_model = clip_model.float()
            input_resolution = clip_model.visual.input_resolution
        
        cdim = clip_dim[args.clip_model]
        clip_model.eval()

    train_supervised_loader, train_unsupervised_loader, val_loader  \
        = create_semi_supervised_dataloaders(args,
        train_img_dir='data/MAMI_processed/images/train',
        train_labeled_csv='data/MAMI_processed/train_labeled_ratio-0.3.csv',
        train_unlabeled_csv='data/MAMI_processed/train_unlabeled_ratio-0.3.csv',
        val_img_dir = 'data/MAMI_processed/images/val',
        val_csv='data/MAMI_processed/val.csv',
        batch_size=args.batchsize, image_size=256,input_resolution=input_resolution)
    
    test_loader = create_semi_supervised_test_dataloaders(args,
                                                        test_img_dir='data/MAMI_processed/images/test',
                                                        test_csv='data/MAMI_processed/test.csv',
                                                        batch_size=args.batchsize, 
                                                        image_size=256,
                                                        input_resolution=input_resolution)

    dataset = {'train_sup': train_supervised_loader,
                'train_unsup': train_unsupervised_loader,
                'val': val_loader,
                'test': test_loader}


    # model = CmmlModel(args)
    model = CmmlModel(args,clip_model = clip_model,cdim = cdim)
    
    if args.dual_stream:
        model = CmmlModel_v2(args)

    if cuda:
        model = model.cuda()


    savepath_folder = args.savepath+"/"+args.experiment+"/"
    if not os.path.exists(args.savepath):
        os.mkdir(args.savepath)
    if not os.path.exists(savepath_folder):
        os.mkdir(savepath_folder)
    print(args)
    train_supervise_loss = train(args,model, dataset,supervise_epochs = args.epochs,
                                        text_supervise_epochs = args.text_supervise_epochs, 
                                        img_supervise_epochs = args.img_supervise_epochs,
                                        lr_supervise = args.lr_supervise, 
                                        text_lr_supervise = args.text_lr_supervise, 
                                        img_lr_supervise = args.img_lr_supervise,
                                        weight_decay = args.weight_decay, 
                                        batchsize = args.batchsize,
                                        textbatchsize = args.textbatchsize,
                                        imgbatchsize = args.imgbatchsize,
                                        cuda = cuda, savepath = savepath_folder,
                                        lambda1=args.lambda1,lambda2=args.lambda2)


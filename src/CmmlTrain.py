from arguments import get_args 
import os  
import torch 
from data.dataClass import MemotionDatasetForCmml
from model.CmmlLayer import CmmlModel
from model.eman import EMAN 
import torch.optim as optim 
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from torch.utils.data import DataLoader 
import datetime 
import numpy as np  
from test import test_multilabel
from tqdm import tqdm
from loss import focal_binary_cross_entropy,zlpr_loss,AsymmetricLoss,ResampleLoss,EntropyLoss
import torch.nn as nn
import wandb
import random
from matplotlib import pyplot as plt
import torch.nn as nn
from utils.plot_utils import get_confusion_matrix
import clip
import open_clip
from model.clip_info import clip_nms,clip_dim
from model.utils import LARS,adjust_learning_rate,exclude_bias_and_norm
from ORG import GB_estimate 
from copy import deepcopy 
import gc 
# from test import test_singlelabel, e
from data.semi_supervised_data import *
from utils.npy_save import npy_save_txt

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
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,[5,10,15],gamma=0.5)

    if args.cal_sep_class_loss:
        print("================calculate loss for seperate class===================")
        if args.use_bce_loss:
            print("==============use bce loss===============")
            criterion = torch.nn.BCELoss()
        elif args.use_asymmetric_loss:
            print("==============use asymmetric loss===============")
            criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
    elif args.use_bce_loss:
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
    # criterion = torch.nn.CrossEntropyLoss()
    print(model.eval())
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
        epoch_img_loss_train = 0
        epoch_text_loss_train = 0
        epoch_total_loss_train = 0

        if not args.train_supervise_only:
            num_steps = min(len(dataset['train_sup']), len(dataset['train_unsup']))
            data_loaders = zip(dataset['train_sup'], dataset['train_unsup'])
        else:
            num_steps = len(dataset['train_sup'])
            data_loaders = zip(dataset['train_sup'])
        for step, batch  in tqdm(enumerate(data_loaders, start=1)
                        ,desc=f'epoch {epoch}',total=num_steps):
            # print(batch_index)
            if args.train_supervise_only:
                (sup_img, sup_text), sup_label = batch[0]
            else:
                (supbatch, unsupbatch) = batch
                (sup_img, sup_text), sup_label = supbatch
                (unsup_img, unsup_text) = unsupbatch

            if args.use_adjust_lr:
                if num_steps == len(dataset['train_sup']):
                    lr = adjust_learning_rate(args, optimizer, dataset['train_sup'], step)
                else:
                    lr = adjust_learning_rate(args, optimizer, dataset['train_unsup'], step)
                wandb.log({"learning rate/lr":lr})
            elif args.use_step_lr:
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

            supervise_img_xx = supervise_img_xx.float()
            label = label.float()
            supervise_img_xx = Variable(supervise_img_xx).cuda() if cuda else Variable(supervise_img_xx)                  
            label = Variable(label).cuda() if cuda else Variable(label)  
            supervise_imghidden = model.Imgmodel(supervise_img_xx,clip_model=model.clip_model)
            
            if args.use_clip:
                supervise_texthidden = model.Textfeaturemodel(clip_input_ids = supervise_clip_ids,clip_model=model.clip_model)
            elif args.use_bert_embedding:
                supervise_texthidden = model.Textfeaturemodel(x = supervise_text_xx,bert_emb = supervise_bert_xx)
            elif args.use_bert_model:
                supervise_texthidden = model.Textfeaturemodel(input_ids = supervise_input_ids,attn_mask = supervise_attn_mask)
            else:
                supervise_texthidden = model.Textfeaturemodel(x = supervise_text_xx)
            
            if not (args.train_supervise_only and args.use_one_head): 
                supervise_imgpredict = model.Imgpredictmodel(supervise_imghidden)
                supervise_textpredict = model.Textpredictmodel(supervise_texthidden)
            if args.use_deep_weak_attention:
                # print("===============use deep weak attention================")
                supervise_imgk = model.Attentionmodel(supervise_imghidden)
                supervise_textk = model.Attentionmodel(supervise_texthidden)
                modality_attention = []
                modality_attention.append(supervise_imgk)
                modality_attention.append(supervise_textk)
                modality_attention = torch.cat(modality_attention, 1)
                modality_attention = nn.functional.softmax(modality_attention, dim = 1)
                img_attention = torch.zeros(1, len(y))
                img_attention[0] = modality_attention[:,0]
                img_attention = img_attention.t()
                text_attention = torch.zeros(1, len(y))
                text_attention[0] = modality_attention[:,1]
                text_attention = text_attention.t()
                if cuda:
                    img_attention = img_attention.cuda()
                    text_attention = text_attention.cuda()
                if args.weighted_attention_sum_feature:
                    supervise_feature_hidden = img_attention * supervise_imghidden + text_attention * supervise_texthidden
                elif args.use_FMI_modalities:
                    if args.use_deep_weak_attention_for_FMI:
                        if args.use_cross:
                            fmi = torch.einsum('b h, b h -> b h h',(img_attention *supervise_imghidden),(text_attention*supervise_texthidden))
                        elif args.use_align:
                            fmi = torch.einsum('b h, b h -> b h',(img_attention *supervise_imghidden),(text_attention*supervise_texthidden))
                    else:
                        if args.use_cross:
                            fmi = torch.einsum('b h, b h -> b h h',supervise_imghidden,supervise_texthidden)
                        elif args.use_align:
                            fmi = torch.einsum('b h, b h -> b h',(img_attention *supervise_imghidden),(text_attention*supervise_texthidden))
                    supervise_feature_hidden = fmi.flatten(start_dim=1)
            elif args.use_coattention:
                # print("===============use co-attention================")
                supervise_feature_hidden = model.FusionCoattention(supervise_imghidden,supervise_texthidden)
            elif args.use_concat_modalities:
                # print("===============use concat================")
                supervise_feature_hidden = torch.cat((supervise_imghidden, supervise_texthidden), dim=1)

            supervise_predict = model.Predictmodel(supervise_feature_hidden) 
            if args.cal_sep_class_loss and args.use_one_head:
                supervise_predict = [sigmoid(p) for p in supervise_predict]
                totalloss = criterion(supervise_predict[0], label[:,0].unsqueeze(1).float()) + \
                    criterion(supervise_predict[1], label[:,1].unsqueeze(1).float()) + \
                    criterion(supervise_predict[2], label[:,2].unsqueeze(1).float()) + \
                    criterion(supervise_predict[3], label[:,3].unsqueeze(1).float())
            else:
                if args.use_focal_loss or  args.use_bce_loss:
                    supervise_predict = sigmoid(supervise_predict)
                    totalloss = criterion(supervise_predict, label)
                    if not args.use_one_head:
                        supervise_textpredict = sigmoid(supervise_textpredict)
                        supervise_imgpredict = sigmoid(supervise_imgpredict)                
                        imgloss = criterion(supervise_imgpredict, label)
                        textloss = criterion(supervise_textpredict, label)
                elif args.use_zlpr_loss or args.use_asymmetric_loss or args.use_resample_loss:
                    if not args.use_one_head:
                        imgloss = criterion(supervise_imgpredict, label)
                        textloss = criterion(supervise_textpredict, label)
                        supervise_imgpredict_unsharp = sigmoid(supervise_imgpredict) 
                        supervise_textpredict_unsharp = sigmoid(supervise_textpredict) 
                        supervise_imgpredict = sigmoid(args.T*supervise_imgpredict)
                        supervise_textpredict = sigmoid(args.T*supervise_textpredict) 
                        
                        totalloss = criterion(supervise_predict, label)
                        if args.supervise_entropy_minimization:
                            entropy_img = EntropyLoss(supervise_imgpredict_unsharp)
                            entropy_text = EntropyLoss(supervise_textpredict_unsharp)
                            entropy_total = EntropyLoss(sigmoid(supervise_predict))
                    else:
                        totalloss = criterion(supervise_predict, label)
            if args.use_one_head:
                if args.use_vicreg:
                    vicreg_loss = model.ProjectormodelImgText(supervise_imgpredict,supervise_textpredict)
                elif args.use_div_dist_for_one_head:
                    supervise_imgpredict = sigmoid(supervise_imgpredict)
                    supervise_textpredict = sigmoid(supervise_textpredict)
                    div = nn.CosineSimilarity(dim=1)(supervise_imgpredict, supervise_textpredict).mean(axis=0)

            elif args.use_div:
                    
                '''
                Diversity Measure code.
                '''        
                div = nn.CosineSimilarity(dim=1)(supervise_imgpredict,supervise_textpredict).mean(axis=0)

                '''
                Diversity Measure code.
                ''' 
            # print("div: ", div.item(), end='\t')
            # div_arr[batch_index-1] = div.item()
            if args.use_org:
                supervise_loss = modality_weights[1]*imgloss + modality_weights[2]*textloss + modality_weights[0]*totalloss
            elif args.use_org_weights:
                supervise_loss = 0.16314931594158397*imgloss + 0.2173853996481428*textloss + 0.6194652844102732*totalloss + 0.01*div
                if args.supervise_entropy_minimization:
                    supervise_loss += entropy_img+entropy_text+entropy_total
            elif args.use_one_head:
                if args.use_vicreg:
                    supervise_loss = 2.0*totalloss+sum(vicreg_loss)
                elif args.use_div_dist_for_one_head:
                    supervise_loss = 2.0*totalloss + 0.01*div
                else:
                    supervise_loss = totalloss
            elif args.use_div:
                supervise_loss = imgloss + textloss + 2.0*totalloss + 0.01* div
            else:
                supervise_loss = imgloss + textloss + 2.0*totalloss

            epoch_total_loss_train += totalloss.item()
            if not args.use_one_head:
                epoch_img_loss_train += imgloss.item()
                epoch_text_loss_train += textloss.item() 
            
            if not args.train_supervise_only:
                    #=======================================#

                    # ================== UNSUPERVISE =================== # 
                    unsupervise_img_xx = unsup_img
                    if args.use_sentence_vectorizer:
                        unsupervise_text_xx = unsup_text['sentence_vectors'].float()
                        unsupervise_text_xx = Variable(unsupervise_text_xx).cuda() if cuda else Variable(unsupervise_text_xx) 

                    if args.use_bert_embedding:
                        unsupervise_bert_xx = unsup_text['sbert_embedding']

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
                    unsupervise_img_xx = Variable(unsupervise_img_xx).cuda() if cuda else Variable(unsupervise_img_xx)     

                    unsupervise_imghidden = model.Imgmodel(unsupervise_img_xx,clip_model=model.clip_model)
                    if args.use_clip:
                        unsupervise_texthidden = model.Textfeaturemodel(clip_input_ids = unsupervise_clip_ids,clip_model=model.clip_model)
                    elif args.use_bert_embedding:
                        unsupervise_texthidden = model.Textfeaturemodel(x = unsupervise_text_xx,bert_emb = unsupervise_bert_xx)
                    elif args.use_bert_model:
                        unsupervise_texthidden = model.Textfeaturemodel(input_ids = unsupervise_token_xx,bert_emb = unsupervise_attn_mask_xx)
                    else:
                        unsupervise_texthidden = model.Textfeaturemodel(x = unsupervise_text_xx)

                    if args.use_one_head:
                        if args.use_vicreg:
                            # print("===============use vcreg in unsupervised data================")
                            vicreg_loss = model.ProjectormodelImgText(model.Imgpredictmodel(unsupervise_imghidden),
                                model.Textpredictmodel(unsupervise_texthidden))
                            total_loss = supervise_loss+sum(vicreg_loss)
                        elif args.use_div_dist_for_one_head:
                            unsupervise_imgpredict = sigmoid(model.Imgpredictmodel(unsupervise_imghidden))
                            unsupervise_textpredict = sigmoid(model.Textpredictmodel(unsupervise_texthidden))
                            dis = 2 - nn.CosineSimilarity(dim=1)(unsupervise_imgpredict, unsupervise_textpredict)
                            tensor1 = dis[torch.abs(dis) < cita]
                            tensor2 = dis[torch.abs(dis) >= cita]
                            tensor1loss = torch.sum(tensor1 * tensor1/2)
                            tensor2loss = torch.sum(cita * (torch.abs(tensor2) - 1/2 * cita))

                            unsupervise_loss = (tensor1loss + tensor2loss)/unsupervise_img_xx.size()[0]  
                            total_loss = supervise_loss +  unsupervise_loss
                            epoch_supervise_loss_train += supervise_loss.item()
                            epoch_div_train += div.item() 
                            epoch_unsupervise_loss_train += unsupervise_loss.item()
                    else:                 
                        unsupervise_imgpredict = model.Imgpredictmodel(unsupervise_imghidden)
                        unsupervise_textpredict = model.Textpredictmodel(unsupervise_texthidden)
                        unsupervise_imgpredict_unsharp = sigmoid(unsupervise_imgpredict)
                        unsupervise_textpredict_unsharp = sigmoid(unsupervise_textpredict)
                        unsupervise_imgpredict = sigmoid(args.T*unsupervise_imgpredict)
                        unsupervise_textpredict = sigmoid(args.T*unsupervise_textpredict)
                        if args.unsupervise_entropy_minimization:
                            unsup_entropy_img = EntropyLoss(unsupervise_imgpredict_unsharp)
                            unsup_entropy_text = EntropyLoss(unsupervise_textpredict_unsharp)

                        '''
                        Robust Consistency Measure code.
                        '''
                        dis = 2 - nn.CosineSimilarity(dim=1)(unsupervise_imgpredict, unsupervise_textpredict)

                        tensor1 = dis[torch.abs(dis) < cita]
                        tensor2 = dis[torch.abs(dis) >= cita]
                        tensor1loss = torch.sum(tensor1 * tensor1/2)
                        tensor2loss = torch.sum(cita * (torch.abs(tensor2) - 1/2 * cita))

                        unsupervise_loss = (tensor1loss + tensor2loss)/unsupervise_img_xx.size()[0]      
                        '''
                        Robust Consistency Measure code.
                        '''
                        if args.unsupervise_entropy_minimization:
                            unsupervise_loss += unsup_entropy_img+unsup_entropy_text
                        # print("unsup loss: ", unsupervise_loss.item())

                        total_loss = supervise_loss +  unsupervise_loss

                        
                    
                        epoch_supervise_loss_train += supervise_loss.item()
                        epoch_div_train += div.item() 
                        epoch_unsupervise_loss_train += unsupervise_loss.item()

                    # ================== UNSUPERVISE =================== #
            else:
                total_loss = supervise_loss
            loss += total_loss.item()
            optimizer.zero_grad()
            total_loss.backward()
            if args.use_clip_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if args.use_multi_step_lr or args.use_linear_scheduler:
            scheduler.step()

        #===================== Multilabel =================#
        if args.use_one_head:
            (f1_macro_multi_total,f1_weighted_multi_total,auc_pm1,
            total_predict, truth,humour,sarcasm,offensive,motivational,humour_truth,
            sarcasm_truth,offensive_truth,motivational_truth) = test_multilabel(args,model.Textfeaturemodel,
            model.Imgpredictmodel, model.Textpredictmodel, model.Imgmodel,
            model.Predictmodel, model.Attentionmodel, dataset['val'], batchsize = batchsize, cuda = cuda,
            FusionCoattention = model.FusionCoattention,clip_model = model.clip_model)
        else:
            (f1_macro_multi_total, f1_macro_multi_img, f1_macro_multi_text,
            f1_weighted_multi_total,f1_weighted_multi_img,f1_weighted_multi_text,
            auc_pm1,auc_pm2,auc_pm3,
            total_predict, truth,humour,sarcasm,offensive,motivational,humour_truth,
            sarcasm_truth,offensive_truth,motivational_truth) = test_multilabel(args,model.Textfeaturemodel,
            model.Imgpredictmodel, model.Textpredictmodel, model.Imgmodel,
            model.Predictmodel, model.Attentionmodel, dataset['val'], batchsize = batchsize, cuda = cuda,
            FusionCoattention = model.FusionCoattention,clip_model = model.clip_model)

        total_step = num_steps
        epoch_supervise_loss_train = epoch_supervise_loss_train/total_step
        epoch_div_train = epoch_div_train/total_step
        epoch_unsupervise_loss_train = epoch_unsupervise_loss_train/total_step
        epoch_img_loss_train = epoch_img_loss_train/total_step
        epoch_text_loss_train = epoch_text_loss_train/total_step 
        epoch_total_loss_train = epoch_total_loss_train/total_step


        print("epoch_total_loss_train:",epoch_total_loss_train)
        if not args.use_one_head:
            print("epoch_img_loss_train",epoch_img_loss_train,
                "\t epoch_text_loss_train:",epoch_text_loss_train,
                "\t epoch_total_loss_train:",epoch_total_loss_train)
        
            print("epoch_supervise_loss_train:",epoch_supervise_loss_train,
            '\t epoch_div_train:',epoch_div_train,'\t epoch_unsupervise_loss_train',epoch_unsupervise_loss_train)

        if not args.use_adjust_lr:
            wandb.log({"learning rate/lr":scheduler.get_last_lr()[0]})

        wandb.log({"f1_macro_multi_total":f1_macro_multi_total})
        if not args.use_one_head:
            wandb.log({"f1_macro_multi_img":f1_macro_multi_img})
            wandb.log({"f1_macro_multi_text":f1_macro_multi_text})

        wandb.log({"f1_weighted_multi_total":f1_weighted_multi_total})
        if not args.use_one_head:
            wandb.log({"f1_weighted_multi_img":f1_weighted_multi_img})
            wandb.log({"f1_weighted_multi_text":f1_weighted_multi_text})
        
        if args.use_one_head:
            print(f"[F1 Macro multilabel] Total: {f1_macro_multi_total}")
            print(f"[F1 weight multilabel] Total: {f1_weighted_multi_total}")
        else:
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

        if not args.use_one_head:
            print('rocauc_pm:    ', auc_pm1,'\t', auc_pm2,'\t', auc_pm3)
        else:
            print('rocauc_pm:    ', auc_pm1)

        # if not args.use_one_head: 
        #     (f1_macro_multi_total, f1_macro_multi_img, f1_macro_multi_text,
        #     f1_weighted_multi_total,f1_weighted_multi_img,f1_weighted_multi_text,
        #     auc_pm1,auc_pm2,auc_pm3,
        #     total_predict, truth,humour,sarcasm,offensive,motivational,humour_truth,
        #     sarcasm_truth,offensive_truth,motivational_truth) = test_multilabel(args,model.Textfeaturemodel,
        #     model.Imgpredictmodel, model.Textpredictmodel, model.Imgmodel,
        #     model.Predictmodel, model.Attentionmodel, dataset['test'], batchsize = batchsize, cuda = cuda)
        # else:
        #     (f1_macro_multi_total,f1_weighted_multi_total,auc_pm1,
        #     total_predict, truth, humour,sarcasm,offensive,motivational,humour_truth,
        #     sarcasm_truth,offensive_truth,motivational_truth) = test_multilabel(args,model.Textfeaturemodel,
        #     None,None, model.Imgmodel,
        #     model.Predictmodel, model.Attentionmodel, dataset['test'], batchsize = batchsize, cuda = cuda)

        # wandb.log({"f1_macro_multi_total_test":f1_macro_multi_total})
        # if not args.use_one_head:
        #     wandb.log({"f1_macro_multi_img_test":f1_macro_multi_img})
        #     wandb.log({"f1_macro_multi_text_test":f1_macro_multi_text})

        # wandb.log({"f1_weighted_multi_total_test":f1_weighted_multi_total})
        # if not args.use_one_head:
        #     wandb.log({"f1_weighted_multi_img_test":f1_weighted_multi_img})
        #     wandb.log({"f1_weighted_multi_text_test":f1_weighted_multi_text})
        
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

    # wandb.init(project="meme_experiments", entity="meme-analysts", mode="disabled")
    wandb.login(key = '9e4b1fe5c252b557ca7eaceff3a78cf738db115e')
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
        if args.unfreeze:
            clip_model.train()
        else:
            clip_model.eval()
    if args.train_supervise_only:
        train_supervise_path = '/home/fptu/viet/SSLMemes/multimodal-misogyny-detection-mami-2022/MAMI_processed/train.csv'
    else:
        train_supervise_path = '/home/fptu/viet/SSLMemes/data/MAMI_processed/train_labeled_ratio-0.3.csv'
        
    train_supervised_loader, train_unsupervised_loader, val_loader  \
        = create_semi_supervised_dataloaders(args,
        train_img_dir='data/MAMI_processed/images/train',
        train_labeled_csv=train_supervise_path,
        train_unlabeled_csv='/home/fptu/viet/SSLMemes/data/MAMI_processed/train_unlabeled_ratio-0.3.csv',
        val_img_dir = 'data/MAMI_processed/images/test',
        val_csv='data/MAMI_processed/test.csv',
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


    model = CmmlModel(args,clip_model = clip_model,cdim = cdim)

    if cuda:
        model = model.cuda()

    savepath_folder = args.savepath+"/"+args.experiment+"/"
    if not os.path.exists(args.savepath):
        os.mkdir(args.savepath)
    if not os.path.exists(savepath_folder):
        os.mkdir(savepath_folder)
    print('Attention module:',model.Attentionmodel.eval())
    print('Predict combine module:',model.Predictmodel.eval())
    print("Predict sub head module:",model.Imgpredictmodel.eval())
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


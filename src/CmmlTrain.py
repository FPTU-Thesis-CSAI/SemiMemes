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
from loss import focal_binary_cross_entropy,zlpr_loss,AsymmetricLoss,ResampleLoss 
import torch.nn as nn
import wandb
import random
from matplotlib import pyplot as plt
import torch.nn as nn
from utils.plot_utils import get_confusion_matrix
from torch.cuda.amp import autocast, GradScaler
import clip
from model.clip_info import clip_nms,clip_dim

# from test import test_singlelabel, e
from data.semi_supervised_data import *
from utils.npy_save import npy_save_txt


def train(args,model, dataset,
        supervise_epochs = 200, text_supervise_epochs = 50, img_supervise_epochs = 50, 
        lr_supervise = 0.01, text_lr_supervise = 0.0001, img_lr_supervise = 0.0001,
        weight_decay = 0, batchsize = 32,lambda1=0.01,lambda2=1, textbatchsize = 32,
        imgbatchsize = 32, cuda = False, savepath = '',eman=None): 
    
    # wandb.watch(model, log="all", log_freq=100)
    model.train()
    print("train")
    # par = []
    # par.append({'params': model.Imgmodel.parameters()})
    # par.append({'params': model.Imgpredictmodel.parameters()})
    # optimizer = optim.Adam(par, lr = img_lr_supervise, weight_decay = weight_decay)
    # scheduler = StepLR(optimizer, step_size = 500, gamma = 0.9) 
    # criterion = torch.nn.BCELoss()
    # train_img_supervise_loss = []
    # batch_count = 0
    loss = 0
    cita = 1.003
    # print("Pretrain img supervise data :")  
    # for epoch in range(1, img_supervise_epochs + 1):
    #     loss = 0
    #     data_loader = DataLoader(dataset = dataset.supervise_(), batch_size = imgbatchsize, shuffle = True, num_workers = 0)
    #     for batch_index, (x, y) in enumerate(data_loader, 1):
    #         batch_count += 1
    #         scheduler.step()
    #         img_xx = x[0]
    #         label = y
    #         img_xx = img_xx.float()
    #         label = label.float()
    #         img_xx = Variable(img_xx).cuda() if cuda else Variable(img_xx)  
    #         label = Variable(label).cuda() if cuda else Variable(label)  
    #         imgxx = model.Imgmodel(img_xx)
    #         imgyy = model.Imgpredictmodel(imgxx)
    #         if args.use_focal_loss:
    #             img_supervise_batch_loss = focal_binary_cross_entropy(args,imgyy, label)
    #         else:
    #             img_supervise_batch_loss = criterion(imgyy, label)
    #         loss += img_supervise_batch_loss.data.item()
    #         optimizer.zero_grad()
    #         img_supervise_batch_loss.backward()
    #         optimizer.step()
    #     print("epoch img loss:",loss/len(data_loader))
    #     if epoch % 1 == 0:
    #         filename = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
    #         torch.save(model.Imgmodel, savepath + filename + 'pretrainimgfeature.pkl')
    #         torch.save(model.Imgpredictmodel, savepath + filename + 'pretrainimgpredict.pkl')
    #         np.save(savepath + filename + "imgsuperviseloss.npy", train_img_supervise_loss)
    #         acc = Imgtest(model.Imgmodel, model.Imgpredictmodel, dataset.test_(), batchsize = imgbatchsize, cuda = cuda)
    #         print('Image supervise - acc :', acc)
    #         print()
    #         # log_image().info(f'----------Img supervise - Accuracy: {acc} --------------------------')  
    #         # np.save(savepath + filename + "imgsuperviseacc.npy", [acc])
    # '''
    # pretrain TextNet.
    # ''' 
    # par = []
    # par.append({'params': model.Textfeaturemodel.parameters()})
    # par.append({'params': model.Textpredictmodel.parameters()})
    # optimizer = optim.Adam(par, lr = text_lr_supervise, weight_decay = weight_decay)
    # scheduler = StepLR(optimizer, step_size = 500, gamma = 0.9) 
    # criterion = torch.nn.BCELoss()
    # train_text_supervise_loss = []
    # batch_count = 0
    # if args.use_bert_model:
    #     if args.freeze_bert_layer_count:
    #         for _, pp in model.Textfeaturemodel.encoder.named_parameters():
    #             pp.requires_grad = False

    #     if args.freeze_bert_layer_count >= 0:
    #         num_hidden_layers = model.Textfeaturemodel.encoder.config.num_hidden_layers
        
    #         layer_idx = [num_hidden_layers-1-i for i in range(args.freeze_bert_layer_count)]
    #         layer_names = ['encoder.layer.{}'.format(j) for j in layer_idx]
    #         for pn, pp in model.Textfeaturemodel.encoder.named_parameters():
    #             if any([ln in pn for ln in layer_names]) or 'pooler.' in pn:
    #                 pp.data = torch.randn(pp.shape)*0.02
    #                 pp.requires_grad = True

    # print('Pretrain text supervise data:')
    # for epoch in range(1, text_supervise_epochs + 1):
    #     loss = 0
    #     data_loader = DataLoader(dataset = dataset.supervise_(), batch_size = textbatchsize, shuffle = True, num_workers = 0)
    #     for batch_index, (x, y) in enumerate(data_loader, 1):
    #         batch_count += 1
    #         scheduler.step()
    #         if args.use_bert_model:
    #             token_xx = x[1]
    #             attn_mask_xx = x[2]
    #             token_xx = token_xx.long()
    #             attn_mask_xx = attn_mask_xx.long()
    #             token_xx = Variable(token_xx).cuda() if cuda else Variable(token_xx) 
    #             attn_mask_xx = Variable(attn_mask_xx).cuda() if cuda else Variable(attn_mask_xx) 
    #         else:
    #             text_xx = x[1]
    #             text_xx = text_xx.float()
    #             text_xx = Variable(text_xx).cuda() if cuda else Variable(text_xx)  
    #             if args.use_bert_embedding:
    #                 bert_xx = x[2]
    #                 bert_xx = bert_xx.float()
    #                 bert_xx = Variable(bert_xx).cuda() if cuda else Variable(bert_xx) 
    #         label = y
    #         label = label.float()                 
    #         label = Variable(label).cuda() if cuda else Variable(label)  
    #         if args.use_bert_embedding:
    #             textxx = model.Textfeaturemodel(x = text_xx,bert_emb = bert_xx)
    #         elif args.use_bert_model:
    #             textxx = model.Textfeaturemodel(input_ids = token_xx,attn_mask = attn_mask_xx)
    #         else:
    #             textxx = model.Textfeaturemodel(x = text_xx)
    #         textyy = model.Textpredictmodel(textxx)
    #         if args.use_focal_loss:
    #             text_supervise_batch_loss = focal_binary_cross_entropy(args,textyy, label)
    #         else:
    #             text_supervise_batch_loss = criterion(textyy, label)
    #         loss += text_supervise_batch_loss.data.item()
    #         optimizer.zero_grad()
    #         text_supervise_batch_loss.backward()
    #         optimizer.step()
    #     print("epoch txt loss:",loss/len(data_loader))
    #     if epoch % text_supervise_epochs == 0:
    #         filename = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
    #         torch.save(model.Textfeaturemodel, savepath + filename + 'pretraintextfeature.pkl')
    #         torch.save(model.Textpredictmodel, savepath + filename + 'pretraintextpredict.pkl')
    #         np.save(savepath + filename + "textsuperviseloss.npy", train_text_supervise_loss)
    #         acc = texttest(args,model.Textfeaturemodel,model.Textpredictmodel, dataset.test_(), batchsize = textbatchsize, cuda = cuda)
    #         print('Text supervise - acc :', acc)
    #         print()
    #         # np.save(savepath + filename + "textsuperviseacc.npy", [acc])
    if args.use_vicreg_pretrain:
        optimizer = optim.Adam(model.parameters(), lr = lr_supervise, weight_decay = weight_decay)
        scheduler = StepLR(optimizer, step_size = 500, gamma = 0.9)  
        pretrain_epoch = supervise_epochs 
        for epoch in range(1, pretrain_epoch + 1):
            total_loss = 0
            num_steps = min(len(dataset['train_sup']), len(dataset['train_unsup']))
            for batch_index, (supbatch, unsupbatch) in tqdm(enumerate(zip(dataset['train_sup'], dataset['train_unsup']), start=1),
                                                    total=min(len(dataset['train_sup']), len(dataset['train_unsup']))):
                (sup_img, sup_text), sup_label = supbatch
                (unsup_img, unsup_text) = unsupbatch   
                scheduler.step()
                supervise_img_xx = sup_img
                if args.use_sentence_vectorizer:
                    supervise_text_xx = sup_text['sentence_vectors'].float()
                    supervise_text_xx = Variable(supervise_text_xx).cuda() if cuda else Variable(supervise_text_xx)  
                if args.use_bert_embedding:
                    supervise_bert_xx = sup_text['sbert_embedding']
                label = sup_label

                if args.use_bert_model:
                    supervise_input_ids = sup_text['input_ids']
                    supervise_attn_mask = sup_text['attention_mask']
                    supervise_input_ids = supervise_input_ids.long()
                    supervise_attn_mask = supervise_attn_mask.long()
                    supervise_input_ids = Variable(supervise_input_ids).cuda() if cuda else Variable(supervise_input_ids)
                    supervise_attn_mask = Variable(supervise_attn_mask).cuda() if cuda else Variable(supervise_attn_mask)

                if args.use_clip:
                    supervise_clip_ids = sup_text['clip_tokens']
                    supervise_clip_ids = Variable(supervise_clip_ids).cuda() if cuda else Variable(supervise_clip_ids)    
                
                supervise_img_xx = supervise_img_xx.float()
                label = label.float()
                supervise_img_xx = Variable(supervise_img_xx).cuda() if cuda else Variable(supervise_img_xx)                  
                label = Variable(label).cuda() if cuda else Variable(label)  
                supervise_imghidden = model.Imgmodel(supervise_img_xx)
                
                if args.use_clip:
                    supervise_texthidden = model.Textfeaturemodel(clip_input_ids = supervise_clip_ids)
                elif args.use_bert_embedding:
                    supervise_texthidden = model.Textfeaturemodel(x = supervise_text_xx,bert_emb = supervise_bert_xx)
                elif args.use_bert_model:
                    supervise_texthidden = model.Textfeaturemodel(input_ids = supervise_input_ids,attn_mask = supervise_attn_mask)
                else:
                    supervise_texthidden = model.Textfeaturemodel(x = supervise_text_xx)
                
                vcreg_loss_supervise = model.Projectormodel(supervise_imghidden,supervise_texthidden)

                unsupervise_img_xx = unsup_img
                if args.use_clip:
                    unsupervise_clip_ids = unsup_text['clip_tokens']
                    unsupervise_clip_ids = Variable(unsupervise_clip_ids).cuda() if cuda else Variable(unsupervise_clip_ids)    

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

                unsupervise_img_xx = unsupervise_img_xx.float()
                unsupervise_img_xx = Variable(unsupervise_img_xx).cuda() if cuda else Variable(unsupervise_img_xx)     

                unsupervise_imghidden = model.Imgmodel(unsupervise_img_xx)
                if args.use_clip:
                    unsupervise_texthidden = model.Textfeaturemodel(clip_input_ids = unsupervise_clip_ids)
                elif args.use_bert_embedding:
                    unsupervise_texthidden = model.Textfeaturemodel(x = unsupervise_text_xx,bert_emb = unsupervise_bert_xx)
                elif args.use_bert_model:
                    unsupervise_texthidden = model.Textfeaturemodel(input_ids = unsupervise_token_xx,bert_emb = unsupervise_attn_mask_xx)
                else:
                    unsupervise_texthidden = model.Textfeaturemodel(x = unsupervise_text_xx)
                vcreg_loss_unsupervise = model.Projectormodel(unsupervise_imghidden,unsupervise_texthidden)
                loss = sum(vcreg_loss_supervise)+sum(vcreg_loss_unsupervise)
                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print("average pretrain loss epoch:",total_loss/num_steps)

    optimizer = optim.Adam(model.parameters(), lr = lr_supervise, weight_decay = weight_decay)
    scheduler = StepLR(optimizer, step_size = 500, gamma = 0.9)  
    if args.use_bce_loss:
        criterion = torch.nn.BCELoss()
    elif args.use_focal_loss:
        criterion = partial(focal_binary_cross_entropy,args)
    elif args.use_zlpr_loss:
        criterion = zlpr_loss
    elif args.use_asymmetric_loss:
        criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
    elif args.use_resample_loss:
        criterion = ResampleLoss(args)
    # criterion = torch.nn.CrossEntropyLoss()
    sigmoid = torch.nn.Sigmoid()
    for epoch in range(1, supervise_epochs + 1):
        print('train multimodal data:', epoch)
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

        # data_loader = DataLoader(dataset = dataset.unsupervise_(), batch_size = batchsize, shuffle = True, num_workers = 0)
        # for batch_index, (x, y) in tqdm(enumerate(data_loader, 1)):

        num_steps = min(len(dataset['train_sup']), len(dataset['train_unsup']))

        # div_arr = np.zeros(shape=(num_steps))
        # img_loss_arr = np.zeros(shape=(num_steps))
        # text_loss_arr = np.zeros(shape=(num_steps))
        # total_suploss_arr = np.zeros(shape=(num_steps))
        # total_final_loss_arr = np.zeros(shape=(num_steps))

        # total_dis = np.zeros(shape=(num_steps, 28))

        for batch_index, (supbatch, unsupbatch) in tqdm(enumerate(zip(dataset['train_sup'], dataset['train_unsup']), start=1),
                                                total=min(len(dataset['train_sup']), len(dataset['train_unsup']))):
            # print(batch_index)

            (sup_img, sup_text), sup_label = supbatch
            (unsup_img, unsup_text) = unsupbatch

            # for single label
            # sup_label = sup_label.unsqueeze(-1) if len(sup_label.shape) == 1 else sup_label # expand last dim for single label only
            # sup_label = torch.stack([1-sup_label, sup_label], axis=-1)
            # for single label

            scheduler.step()
            # x[0] = torch.cat(x[0], 0)
            # x[1] = torch.cat(x[1], 0)
            # if args.use_bert_embedding:
            #     x[2] = torch.cat(x[2], 0)
            y = sup_label
            '''
            Attention architecture and use bceloss.
            '''
            supervise_img_xx = sup_img
            if args.use_sentence_vectorizer:
                supervise_text_xx = sup_text['sentence_vectors'].float()
                supervise_text_xx = Variable(supervise_text_xx).cuda() if cuda else Variable(supervise_text_xx)  
            if args.use_bert_embedding:
                supervise_bert_xx = sup_text['sbert_embedding']
            label = sup_label

            if args.use_bert_model:
                supervise_input_ids = sup_text['input_ids']
                supervise_attn_mask = sup_text['attention_mask']
                supervise_input_ids = supervise_input_ids.long()
                supervise_attn_mask = supervise_attn_mask.long()
                supervise_input_ids = Variable(supervise_input_ids).cuda() if cuda else Variable(supervise_input_ids)
                supervise_attn_mask = Variable(supervise_attn_mask).cuda() if cuda else Variable(supervise_attn_mask)

            if args.use_clip:
                supervise_clip_ids = sup_text['clip_tokens']
                supervise_clip_ids = Variable(supervise_clip_ids).cuda() if cuda else Variable(supervise_clip_ids)    

            supervise_img_xx = supervise_img_xx.float()
            label = label.float()
            supervise_img_xx = Variable(supervise_img_xx).cuda() if cuda else Variable(supervise_img_xx)                  
            label = Variable(label).cuda() if cuda else Variable(label)  
            supervise_imghidden = model.Imgmodel(supervise_img_xx)
            
            if args.use_clip:
                supervise_texthidden = model.Textfeaturemodel(clip_input_ids = supervise_clip_ids)
            elif args.use_bert_embedding:
                supervise_texthidden = model.Textfeaturemodel(x = supervise_text_xx,bert_emb = supervise_bert_xx)
            elif args.use_bert_model:
                supervise_texthidden = model.Textfeaturemodel(input_ids = supervise_input_ids,attn_mask = supervise_attn_mask)
            else:
                supervise_texthidden = model.Textfeaturemodel(x = supervise_text_xx)
            
            if args.use_vicreg_in_training:
                vcreg_loss_supervise = model.Projectormodel(supervise_imghidden,supervise_texthidden)
            supervise_imgpredict = model.Imgpredictmodel(supervise_imghidden)
            supervise_textpredict = model.Textpredictmodel(supervise_texthidden)
            
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
            supervise_feature_hidden = img_attention * supervise_imghidden + text_attention * supervise_texthidden
            supervise_predict = model.Predictmodel(supervise_feature_hidden)       
            if args.use_focal_loss or  args.use_bce_loss:
                supervise_textpredict = sigmoid(supervise_textpredict)
                supervise_imgpredict = sigmoid(supervise_imgpredict)
                supervise_predict = sigmoid(supervise_predict)
                totalloss = criterion(supervise_predict, label)
                imgloss = criterion(supervise_imgpredict, label)
                textloss = criterion(supervise_textpredict, label)
            elif args.use_zlpr_loss or args.use_asymmetric_loss or args.use_resample_loss:
                totalloss = criterion(supervise_predict, label)
                imgloss = criterion(supervise_imgpredict, label)
                textloss = criterion(supervise_textpredict, label)
            '''
            Diversity Measure code.
            '''         
            # similar = torch.bmm(supervise_imgpredict.unsqueeze(1), supervise_textpredict.unsqueeze(2)).view(supervise_imgpredict.size()[0])
            # norm_matrix_img = torch.norm(supervise_imgpredict, 2, dim = 1)
            # norm_matrix_text = torch.norm(supervise_textpredict, 2, dim = 1)
            # div = torch.mean(similar/(norm_matrix_img * norm_matrix_text))

            div = nn.CosineSimilarity(dim=1)(supervise_imgpredict, supervise_textpredict).mean(axis=0)

            '''
            Diversity Measure code.
            ''' 
            # print("div: ", div.item(), end='\t')
            # div_arr[batch_index-1] = div.item()

            if args.use_auto_weight:
                supervise_loss = 1/(2*model.Predictmodel.sigma[0]**2)*imgloss + 1/(2*model.Predictmodel.sigma[1]**2)*textloss \
                + 1/(2*model.Predictmodel.sigma[2]**2)*totalloss + torch.log(model.Predictmodel.sigma).sum()
            else:
                supervise_loss = imgloss + textloss + 2.0*totalloss

            # print('img: ', imgloss.item(), ' text: ', textloss.item(), 'total: ', totalloss.item(), end="\t")
            # img_loss_arr[batch_index-1] = imgloss.item()
            # text_loss_arr[batch_index-1] = textloss.item()
            # total_suploss_arr[batch_index-1] = totalloss.item()

            # can not log by wandb
            # wandb.log({"supervise_predict":supervise_predict.detach().cpu().numpy(),
            #             "supervise_imgpredict":supervise_imgpredict.detach().cpu().numpy(),
            #             "supervise_textpredict":supervise_textpredict.detach().cpu().numpy(),
            #             "label":label.detach().cpu().numpy()})
            
            # LOG DATA TO FIX A BUG: /pytorch/aten/src/ATen/native/cuda/Loss.cu:115: operator(): block: [0,0,0], thread: [0,0,0] Assertion `input_val >= zero && input_val <= one` failed.

            #=======================================#

            # ================== UNSUPERVISE =================== # 

            epoch_img_loss_train += imgloss.item()
            epoch_text_loss_train += textloss.item() 
            epoch_total_loss_train += totalloss.item()


            unsupervise_img_xx = unsup_img
            if args.use_sentence_vectorizer:
                unsupervise_text_xx = unsup_text['sentence_vectors'].float()
                unsupervise_text_xx = Variable(unsupervise_text_xx).cuda() if cuda else Variable(unsupervise_text_xx) 

            if args.use_bert_embedding:
                # x[3] = torch.cat(x[3], 0)
                # x[4] = torch.cat(x[4], 0)
                # x[5] = torch.cat(x[5], 0)
                unsupervise_bert_xx = unsup_text['sbert_embedding']
            # else:
            #     x[2] = torch.cat(x[2], 0)
            #     x[3] = torch.cat(x[3], 0)
            #     unsupervise_img_xx = x[2]
            #     unsupervise_text_xx = x[3]
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

            unsupervise_imghidden = model.Imgmodel(unsupervise_img_xx)
            if args.use_clip:
                unsupervise_texthidden = model.Textfeaturemodel(clip_input_ids = unsupervise_clip_ids)
            elif args.use_bert_embedding:
                unsupervise_texthidden = model.Textfeaturemodel(x = unsupervise_text_xx,bert_emb = unsupervise_bert_xx)
            elif args.use_bert_model:
                unsupervise_texthidden = model.Textfeaturemodel(input_ids = unsupervise_token_xx,bert_emb = unsupervise_attn_mask_xx)
            else:
                unsupervise_texthidden = model.Textfeaturemodel(x = unsupervise_text_xx)

            if  args.use_vicreg_in_training:
                vcreg_loss_unsupervise = model.Projectormodel(unsupervise_imghidden,unsupervise_texthidden)

            unsupervise_imgpredict = sigmoid(model.Imgpredictmodel(unsupervise_imghidden))
            unsupervise_textpredict = sigmoid(model.Textpredictmodel(unsupervise_texthidden))

            '''
            Robust Consistency Measure code.
            '''
            # unsimilar = torch.bmm(unsupervise_imgpredict.unsqueeze(1), unsupervise_textpredict.unsqueeze(2)).view(unsupervise_imgpredict.size()[0])
            # unnorm_matrix_img = torch.norm(unsupervise_imgpredict, 2, dim = 1)
            # unnorm_matrix_text = torch.norm(unsupervise_textpredict, 2, dim = 1)
            # dis = 2 - unsimilar/(unnorm_matrix_img * unnorm_matrix_text)

            dis = 2 - nn.CosineSimilarity(dim=1)(unsupervise_imgpredict, unsupervise_textpredict)

            # print("dis: ", dis.detach().cpu().numpy())

            # total_dis[batch_index-1] = dis.detach().cpu().numpy()
            

            tensor1 = dis[torch.abs(dis) < cita]
            tensor2 = dis[torch.abs(dis) >= cita]
            tensor1loss = torch.sum(tensor1 * tensor1/2)
            tensor2loss = torch.sum(cita * (torch.abs(tensor2) - 1/2 * cita))

            unsupervise_loss = (tensor1loss + tensor2loss)/unsupervise_img_xx.size()[0]      
            '''
            Robust Consistency Measure code.
            '''

            # print("unsup loss: ", unsupervise_loss.item())

            if args.use_vicreg_in_training:
                total_loss = supervise_loss + 0.01* div +  unsupervise_loss + sum(vcreg_loss_unsupervise) + sum(vcreg_loss_supervise)
                if args.use_sim_loss:
                    epoch_v_supervise_loss_train += vcreg_loss_supervise[0].item()
                    epoch_c_supervise_loss_train += vcreg_loss_supervise[2].item()
                    epoch_v_unsupervise_loss_train += vcreg_loss_unsupervise[0].item()
                    epoch_c_unsupervise_loss_train += vcreg_loss_unsupervise[2].item()
                    epoch_i_supervise_loss_train += vcreg_loss_unsupervise[1].item()
                    epoch_i_unsupervise_loss_train += vcreg_loss_supervise[1].item()     
                else:
                    epoch_v_supervise_loss_train += vcreg_loss_supervise[0].item()
                    epoch_c_supervise_loss_train += vcreg_loss_supervise[1].item()
                    epoch_v_unsupervise_loss_train += vcreg_loss_unsupervise[0].item()
                    epoch_c_unsupervise_loss_train += vcreg_loss_unsupervise[1].item()
            else:
                total_loss = supervise_loss + 0.01* div +  unsupervise_loss
            epoch_supervise_loss_train += supervise_loss.item()
            epoch_div_train += div.item() 
            epoch_unsupervise_loss_train += unsupervise_loss.item()

            # ================== UNSUPERVISE =================== #
            
            # total_loss = supervise_loss # FOR DEBUG ONLY
            # print("total loss: ", total_loss.item())
            # total_final_loss_arr[batch_index-1] = total_loss.item()

            loss += total_loss.item()
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            if args.use_eman:
                eman.update(model)
        
        if epoch % 10 == 0:
            filename = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
            torch.save(model.Imgmodel, os.path.join(savepath, filename +'img.pkl'))
            torch.save(model.Textfeaturemodel, os.path.join(savepath, filename + 'Textfeaturemodel.pkl'))
            torch.save(model.Imgpredictmodel, os.path.join(savepath, filename + 'Imgpredictmodel.pkl'))
            torch.save(model.Textpredictmodel, os.path.join(savepath, filename + 'Textpredictmodel.pkl'))
            torch.save(model.Attentionmodel, os.path.join(savepath, filename +'attention.pkl'))
        

        #===================== Multilabel =================#
        (f1_macro_multi_total, f1_macro_multi_img, f1_macro_multi_text, total_predict, truth, f1_skl1,
        f1_skl2, f1_skl3, f1_pm1, f1_pm2, f1_pm3,
        auc_pm1,auc_pm2,auc_pm3, acc1, acc2, acc3, 
        coverage1, coverage2, coverage3, example_auc1,
        example_auc2, example_auc3, macro_auc1, macro_auc2,
        macro_auc3, micro_auc1, micro_auc2, micro_auc3,
        ranking_loss1, ranking_loss2, ranking_loss3,
        humour,sarcasm,offensive,motivational,humour_truth,
        sarcasm_truth,offensive_truth,motivational_truth) = test_multilabel(args,model.Textfeaturemodel,
        model.Imgpredictmodel, model.Textpredictmodel, model.Imgmodel,
        model.Predictmodel, model.Attentionmodel, dataset['val'], batchsize = batchsize, cuda = cuda)
        
        # total_step = len(data_loader)
        total_step = num_steps
        epoch_supervise_loss_train = epoch_supervise_loss_train/total_step
        epoch_div_train = epoch_div_train/total_step
        epoch_unsupervise_loss_train = epoch_unsupervise_loss_train/total_step
        epoch_img_loss_train = epoch_img_loss_train/total_step
        epoch_text_loss_train = epoch_text_loss_train/total_step 
        epoch_total_loss_train = epoch_total_loss_train/total_step

        # wandb.log({"train_loss/epoch_supervise_loss_train":epoch_supervise_loss_train})
        # wandb.log({"train_loss/epoch_div_train":epoch_div_train})
        # wandb.log({"train_loss/epoch_unsupervise_loss_train":epoch_unsupervise_loss_train})
        # wandb.log({"train_loss/epoch_img_loss_train":epoch_img_loss_train})
        # wandb.log({"train_loss/epoch_text_loss_train":epoch_text_loss_train})
        # wandb.log({"train_loss/epoch_total_loss_train":epoch_total_loss_train})

        if model.Projectormodel != None:
            epoch_v_supervise_loss_train = epoch_v_supervise_loss_train/total_step
            epoch_c_supervise_loss_train = epoch_c_supervise_loss_train/total_step
            epoch_v_unsupervise_loss_train = epoch_v_unsupervise_loss_train/total_step
            epoch_c_unsupervise_loss_train = epoch_c_unsupervise_loss_train/total_step
            wandb.log({"train_loss/epoch_v_supervise_loss_train":epoch_v_supervise_loss_train})
            wandb.log({"train_loss/epoch_c_supervise_loss_train":epoch_c_supervise_loss_train})
            wandb.log({"train_loss/epoch_v_unsupervise_loss_train":epoch_v_unsupervise_loss_train})
            wandb.log({"train_loss/epoch_c_unsupervise_loss_train":epoch_c_unsupervise_loss_train})
            if args.use_sim_loss:
                epoch_i_supervise_loss_train = epoch_i_supervise_loss_train/total_step
                epoch_i_unsupervise_loss_train = epoch_i_unsupervise_loss_train/total_step
                wandb.log({"train_loss/epoch_i_supervise_loss_train":epoch_i_supervise_loss_train})
                wandb.log({"train_loss/epoch_i_unsupervise_loss_train":epoch_i_unsupervise_loss_train})

        print("epoch_img_loss_train",epoch_img_loss_train,
        "\t epoch_text_loss_train:",epoch_text_loss_train,
        "\t epoch_total_loss_train:",epoch_total_loss_train)
        
        print("epoch_supervise_loss_train:",epoch_supervise_loss_train,
        '\t epoch_div_train:',epoch_div_train,'\t epoch_unsupervise_loss_train',epoch_unsupervise_loss_train)
        if model.Projectormodel != None:
            print("epoch_v_supervise_loss_train:",epoch_v_supervise_loss_train,
            "\t epoch_c_supervise_loss_train:",epoch_c_supervise_loss_train)

            print("epoch_v_unsupervise_loss_train:",epoch_v_unsupervise_loss_train,
            "\t epoch_c_unsupervise_loss_train:",epoch_c_unsupervise_loss_train)
            if args.use_sim_loss:
                print("epoch_i_supervise_loss_train:",epoch_i_supervise_loss_train,
                "\t epoch_i_unsupervise_loss_train:",epoch_i_unsupervise_loss_train)
        print("f1_macro_multi_1:",f1_macro_multi_total,
        "\t f1_macro_multi_2",f1_macro_multi_img,
        "\t f1_macro_multi_3:",f1_macro_multi_text)

        wandb.log({"learning rate/lr":scheduler.get_last_lr()[0]})
        wandb.log({"f1_macro_multi_total":f1_macro_multi_total})
        wandb.log({"f1_macro_multi_img":f1_macro_multi_img})
        wandb.log({"f1_macro_multi_text":f1_macro_multi_text})
        
        print(f"[F1 Macro multilabel] Total: {f1_macro_multi_total} Image {f1_macro_multi_img} Text {f1_macro_multi_text}")

        # wandb.log({"f1_skl_all":f1_skl1})
        # wandb.log({"f1_skl_image":f1_skl2})
        # wandb.log({"f1_skl_text":f1_skl3})

        # wandb.log({"f1_pytorch_all":f1_pm1})
        # wandb.log({"f1_pytorch_image":f1_pm2})
        # wandb.log({"f1_pytorch_text":f1_pm3})

        # wandb.log({"Prediction": total_predict })
        # wandb.log({"Ground_truth": truth })


        # wandb.log({"roc": wandb.plot.roc_curve(truth[:,0], np.expand_dims(total_predict[:,0], axis=-1 ))})
        # wandb.log({"roc": wandb.plot.roc_curve(truth[:,1], np.expand_dims(total_predict[:,1], axis=-1 ) )})
        # wandb.log({"roc": wandb.plot.roc_curve(truth[:,2], np.expand_dims(total_predict[:,2], axis=-1 ) )})
        # wandb.log({"roc": wandb.plot.roc_curve(truth[:,3], np.expand_dims(total_predict[:,3], axis=-1 ) )})

        # wandb.log({f"roc/epoch{epoch}/_roc_humour": wandb.plot.roc_curve(truth[:,0], np.stack([ 1-total_predict[:,0], total_predict[:,0] ], axis=1)  ) })
        # wandb.log({f"roc/epoch{epoch}/_roc_sarcasm": wandb.plot.roc_curve( truth[:,1], np.stack([ 1-total_predict[:,1], total_predict[:,1] ], axis=1) ) })
        # wandb.log({f"roc/epoch{epoch}/_roc_offensive": wandb.plot.roc_curve(truth[:,2], np.stack([ 1-total_predict[:,2], total_predict[:,2] ], axis=1) ) })
        # wandb.log({f"roc/epoch{epoch}/_roc_motivational": wandb.plot.roc_curve(truth[:,3], np.stack([ 1-total_predict[:,3], total_predict[:,3] ], axis=1) ) })

        total_2 = (total_predict > 0.5).astype('int')

        # ['shaming', 'stereotype', 'objectification', 'violence']
        get_confusion_matrix(truth[:,0],total_2[:,0])
        wandb.log({"confusion_matrix_shaming": plt})
        get_confusion_matrix(truth[:,1],total_2[:,1])
        wandb.log({"confusion_matrix_stereotype": plt})
        get_confusion_matrix(truth[:,2],total_2[:,2])
        wandb.log({"confusion_matrix_objectification": plt})
        get_confusion_matrix(truth[:,3],total_2[:,3])
        wandb.log({"confusion_matrix_violence": plt})

        

        # wandb.log({"conf_mat_shaming": wandb.Image(confusion_matrix(truth[:,0],total_2[:,0]))})
        # wandb.log({"conf_mat_stereotype": wandb.Image(confusion_matrix(truth[:,1],total_2[:,1]))})
        # wandb.log({"conf_mat_objectification": wandb.Image(confusion_matrix(truth[:,2],total_2[:,2]))})
        # wandb.log({"conf_mat_violence": wandb.Image(confusion_matrix(truth[:,3],total_2[:,3]))})

        # total_2 = a > 0.5
        # Visualize single plot


        # wandb.log({f"conf_mat/epoch{epoch}/_confmat_humour": wandb.plot.confusion_matrix( y_true = truth[:,0], preds = total_2[:,0], class_names=["not humour", "humour"] )})
        # wandb.log({f"conf_mat/epoch{epoch}/_confmat_sarcasm": wandb.plot.confusion_matrix( y_true = truth[:,1], preds = total_2[:,1], class_names=["not sarcasm", "sarcasm"] )})
        # wandb.log({f"conf_mat/epoch{epoch}/_confmat_offensive": wandb.plot.confusion_matrix( y_true = truth[:,2], preds = total_2[:,2], class_names=["not offensive", "offensive"] )})
        # wandb.log({f"conf_mat/epoch{epoch}/_confmat_motivational": wandb.plot.confusion_matrix( y_true = truth[:,3], preds = total_2[:,3], class_names=["not motivational", "motivational"] )})

        # , labels = ["humour"]
        # , labels = ["sarcasm"]
        # , labels = ["offensive"]
        # , labels = ["motivational"]

        wandb.log({"histogram/_hist_label_shaming_pred":wandb.Histogram(np_histogram = humour)})
        wandb.log({"histogram/_hist_label_stereotype_pred":wandb.Histogram(np_histogram = sarcasm)})
        wandb.log({"histogram/_hist_label_objectification_pred":wandb.Histogram(np_histogram = offensive)})
        wandb.log({"histogram/_hist_label_violence_pred":wandb.Histogram(np_histogram = motivational)})

        wandb.log({"histogram/_hist_label_shaming_truth":wandb.Histogram(np_histogram = humour_truth)})
        wandb.log({"histogram/_hist_label_stereotype_truth":wandb.Histogram(np_histogram = sarcasm_truth)})
        wandb.log({"histogram/_hist_label_objectification_truth":wandb.Histogram(np_histogram = offensive_truth)})
        wandb.log({"histogram/_hist_label_violence_truth":wandb.Histogram(np_histogram = motivational_truth)})


        # print('f1_skl:    ', f1_skl1,'\t', f1_skl2,'\t', f1_skl3)
        # print('f1_pm:    ', f1_pm1,'\t', f1_pm2,'\t', f1_pm3)
        # print('coverage:    ', coverage1,'\t', coverage2,'\t', coverage3)
        print('rocauc_pm:    ', auc_pm1,'\t', auc_pm2,'\t', auc_pm3)
        # print('example_auc: ',  example_auc1,'\t', example_auc2,'\t', example_auc3)
        # print('macro_auc:   ',  macro_auc1,'\t', macro_auc2,'\t', macro_auc3)
        # print('micro_auc:   ',  micro_auc1,'\t', micro_auc2,'\t', micro_auc3)
        # print('ranking_loss:',  ranking_loss1,'\t', ranking_loss2,'\t', ranking_loss3)
        # print()

        #=================== Single label =====================#
        # macro_fbatch_size=args.batchsize, image_size=2561_all,macro_f1_image,macro_f1_text, macro_roc_auc1, macro_roc_auc2, macro_roc_auc3, total_predict, truth, misogynous, misogynous_truth = test_singlelabel(args, model.Textfeaturemodel, model.Imgpredictmodel, model.Textpredictmodel, model.Imgmodel, model.Predictmodel, model.Attentionmodel, dataset['val'], batchsize = batchsize, cuda = cuda)
        
        # #print("------Epoch : ", epoch, "------" )
        # #log_train().info(f'\n --------Epoch :{epoch}--------------')
        # #print("-------------- Total ---------------------- Image -------------------- Text----")
        # #log_train().info(f'-------------- Total ---------------------- Image -------------------- Text----')
        # # print('Acc:         ', acc1,'\t', acc2,'\t', acc3)
        # # log_train().info(f'\n[logger] Acc = \n{acc1},{acc2},{acc3}'
        # #                                 f'\n{"-" * 100}')
        # # log_train().info(f'\n Acc:      {acc1}      {acc2}    {acc3}')
        # # log_train().info(f'\n coverage:      {coverage1}      {coverage2}    {coverage3}')
        # # log_train().info(f'\n example_auc:      {example_auc1}      {example_auc2}    {example_auc3}')
        # # log_train().info(f'\n macro_auc:      {macro_auc1}      {macro_auc2}    {macro_auc3}')
        # # log_train().info(f'\n micro_auc:      {micro_auc1}      {micro_auc2}    {micro_auc3}')
        # # log_train().info(f'\n ranking_loss:      {ranking_loss1}      {ranking_loss2}    {ranking_loss3}')
        # # log_train().info(f'\n --------Loss :{loss}--------------')

        # # wandb.log({"f1_skl_all":f1_skl1})
        # # wandb.log({"f1_skl_image":f1_skl2})
        # # wandb.log({"f1_skl_text":f1_skl3})

        # wandb.log({"macro_roc_auc_single":macro_roc_auc1})
        # wandb.log({"macro_roc_auc_image_single":macro_roc_auc2})
        # wandb.log({"macro_roc_auc_text_single":macro_roc_auc3})

        # # wandb.log({"Prediction": total_predict })
        # # wandb.log({"Ground_truth": truth })
        

        # # wandb.log({f"roc/epoch{epoch}/_roc_misogynous": wandb.plot.roc_curve(truth[:,0], np.stack([ 1-total_predict[:,0], total_predict[:,0] ], axis=1)  ) })


        # total_2 = total_predict > 0.5
        # # wandb.log({f"conf_mat/epoch{epoch}/_confmat_misogynous": wandb.plot.confusion_matrix( y_true = truth[:,0], preds = total_2[:,0], class_names=["not misogynous", "misogynous"] )})
        # # softmax
        # wandb.log({f"conf_mat/epoch{epoch}/_confmat_misogynous": wandb.plot.confusion_matrix( y_true = truth[:,1], preds = total_2[:,1], class_names=["not misogynous", "misogynous"] )})

        # # wandb.log({"histogram/_hist_label_humour_pred":wandb.Histogram(np_histogram = humour)})
        # # wandb.log({"histogram/_hist_label_sarcasm_pred":wandb.Histogram(np_histogram = sarcasm)})
        # # wandb.log({"histogram/_hist_label_offensive_pred":wandb.Histogram(np_histogram = offensive)})
        # # wandb.log({"histogram/_hist_label_motivational_pred":wandb.Histogram(np_histogram = motivational)})

        # # wandb.log({"histogram/_hist_label_humour_truth":wandb.Histogram(np_histogram = humour_truth)})
        # # wandb.log({"histogram/_hist_label_sarcasm_truth":wandb.Histogram(np_histogram = sarcasm_truth)})
        # # wandb.log({"histogram/_hist_label_offensive_truth":wandb.Histogram(np_histogram = offensive_truth)})
        # # wandb.log({"histogram/_hist_label_motivational_truth":wandb.Histogram(np_histogram = motivational_truth)})

        # wandb.log({"macro_f1_all":macro_f1_all})
        # wandb.log({"macro_f1_image":macro_f1_image})
        # wandb.log({"macro_f1_text":macro_f1_text})

        # print('f1_macro:    ', macro_f1_all,'\t', macro_f1_image,'\t', macro_f1_text)
        # # print('f1_pm:    ', f1_pm1,'\t', f1_pm2,'\t', f1_pm3)
        # # print('coverage:    ', coverage1,'\t', coverage2,'\t', coverage3)
        # # print('auc_pm:    ', auc_pm1,'\t', auc_pm2,'\t', auc_pm3)
        # # print('example_auc: ',  example_auc1,'\t', example_auc2,'\t', example_auc3)
        # # print('macro_auc:   ',  macro_auc1,'\t', macro_auc2,'\t', macro_auc3)
        # # print('micro_auc:   ',  micro_auc1,'\t', micro_auc2,'\t', micro_auc3)
        # # print('ranking_loss:',  ranking_loss1,'\t', ranking_loss2,'\t', ranking_loss3)
        # print()
        # #np.save(savepath + filename + "superviseacc.npy", [acc1, acc2, acc3])

        # ============================= TEST ======================= #
        (f1_macro_multi_total, f1_macro_multi_img, f1_macro_multi_text, total_predict, truth, f1_skl1,
        f1_skl2, f1_skl3, f1_pm1, f1_pm2, f1_pm3,
        auc_pm1,auc_pm2,auc_pm3, acc1, acc2, acc3, 
        coverage1, coverage2, coverage3, example_auc1,
        example_auc2, example_auc3, macro_auc1, macro_auc2,
        macro_auc3, micro_auc1, micro_auc2, micro_auc3,
        ranking_loss1, ranking_loss2, ranking_loss3,
        humour,sarcasm,offensive,motivational,humour_truth,
        sarcasm_truth,offensive_truth,motivational_truth) = test_multilabel(args,model.Textfeaturemodel,
        model.Imgpredictmodel, model.Textpredictmodel, model.Imgmodel,
        model.Predictmodel, model.Attentionmodel, dataset['test'], batchsize = batchsize, cuda = cuda)

        wandb.log({"f1_macro_multi_total_test":f1_macro_multi_total})
        wandb.log({"f1_macro_multi_img_test":f1_macro_multi_img})
        wandb.log({"f1_macro_multi_text_test":f1_macro_multi_text})
        
        print(f"Test [F1 Macro multilabel] Total: {f1_macro_multi_total} Image {f1_macro_multi_img} Text {f1_macro_multi_text}")
        
        #=====================EMAN Multilabel =================#
        if args.use_eman:
            (f1_macro_multi_total, f1_macro_multi_img, f1_macro_multi_text, total_predict, truth, f1_skl1,
            f1_skl2, f1_skl3, f1_pm1, f1_pm2, f1_pm3,
            auc_pm1,auc_pm2,auc_pm3, acc1, acc2, acc3, 
            coverage1, coverage2, coverage3, example_auc1,
            example_auc2, example_auc3, macro_auc1, macro_auc2,
            macro_auc3, micro_auc1, micro_auc2, micro_auc3,
            ranking_loss1, ranking_loss2, ranking_loss3,
            humour,sarcasm,offensive,motivational,humour_truth,
            sarcasm_truth,offensive_truth,motivational_truth) = test_multilabel(args,eman.module.Textfeaturemodel,
            eman.module.Imgpredictmodel, eman.module.Textpredictmodel, eman.module.Imgmodel,
            eman.module.Predictmodel, eman.module.Attentionmodel, dataset['val'], batchsize = batchsize, cuda = cuda)
            
            print(f"Val [F1 Macro multilabel] Total for EMAN: {f1_macro_multi_total} Image {f1_macro_multi_img} Text {f1_macro_multi_text}")
            
            (f1_macro_multi_total, f1_macro_multi_img, f1_macro_multi_text, total_predict, truth, f1_skl1,
            f1_skl2, f1_skl3, f1_pm1, f1_pm2, f1_pm3,
            auc_pm1,auc_pm2,auc_pm3, acc1, acc2, acc3, 
            coverage1, coverage2, coverage3, example_auc1,
            example_auc2, example_auc3, macro_auc1, macro_auc2,
            macro_auc3, micro_auc1, micro_auc2, micro_auc3,
            ranking_loss1, ranking_loss2, ranking_loss3,
            humour,sarcasm,offensive,motivational,humour_truth,
            sarcasm_truth,offensive_truth,motivational_truth) = test_multilabel(args,eman.module.Textfeaturemodel,
            eman.module.Imgpredictmodel, eman.module.Textpredictmodel, eman.module.Imgmodel,
            eman.module.Predictmodel, eman.module.Attentionmodel, dataset['test'], batchsize = batchsize, cuda = cuda)

            print(f"Test [F1 Macro multilabel] Total for EMAN: {f1_macro_multi_total} Image {f1_macro_multi_img} Text {f1_macro_multi_text}")
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
    wandb.init(project="meme_experiments", entity="meme-analysts")
    # wandb.init()

    wandb.run.name = args.experiment
    print(f"Experiement: {wandb.run.name}")

    if args.use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpu
        cuda = torch.cuda.is_available() and args.use_gpu
    else:
        cuda = False

    # dataset = MemotionDatasetForCmml(args,args.imgfilenamerecord, 
    #                     args.imgfilenamerecord_unlabel, 
    #                     args.imgfilename, args.textfilename, 
    #                     args.textfilename_unlabel, 
    #                     args.labelfilename, 
    #                     args.labelfilename_unlabel, 
    #                     args.imgfilenamerecord_val, 
    #                     args.imgfilename_val, 
    #                     args.textfilename_val, 
    #                     args.labelfilename_val,
    #                     args.sbertemb,
    #                     args.sbertemb_unlabel,
    #                     args.sbertemb_val,
    #                     train = True, 
    #                     supervise = True)
    input_resolution = None
    clip_model = None
    cdim = None
    if args.use_clip:
        clip_model, _ = clip.load(clip_nms[args.vmodel],jit=False)
        clip_model = clip_model.float()
        input_resolution = clip_model.visual.input_resolution
        cdim = clip_dim[args.vmodel]
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


    model = CmmlModel(args,clip_model = clip_model,cdim = cdim)
    eman=None 
    if args.use_eman:
        eman = EMAN(model, 0.9997)

    if cuda:
        model = model.cuda()
        if args.use_eman:
            eman = eman.cuda()

    savepath_folder = args.savepath+"/"+args.experiment+"/"
    if not os.path.exists(args.savepath):
        os.mkdir(args.savepath)
    if not os.path.exists(savepath_folder):
        os.mkdir(savepath_folder)
    print(model.eval())
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
                                    lambda1=args.lambda1,lambda2=args.lambda2,eman=eman)



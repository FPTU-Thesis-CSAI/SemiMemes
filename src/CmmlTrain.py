from arguments import get_args 
import os  
import torch 
from data.dataClass import MemotionDatasetForCmml
from model.CmmlLayer import CmmlModel
import torch.optim as optim 
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from torch.utils.data import DataLoader 
import datetime 
import numpy as np  
from test import *
from tqdm import tqdm
from loss import focal_binary_cross_entropy 
import torch.nn as nn
import wandb
import random

from test import test_singlelabel
from data.semi_supervised_data import *



def train(args,model, dataset,
          supervise_epochs = 200, text_supervise_epochs = 50, img_supervise_epochs = 50, 
          lr_supervise = 0.01, text_lr_supervise = 0.0001, img_lr_supervise = 0.0001,
          weight_decay = 0, batchsize = 32,lambda1=0.01,lambda2=1, textbatchsize = 32,
           imgbatchsize = 32, cuda = False, savepath = ''): 
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
    # loss_batch = 50
    # print("Pretrain img supervise data :")  
    # for epoch in range(1, img_supervise_epochs + 1):
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
    #         if batch_count >= loss_batch:
    #             loss = loss/loss_batch
    #             train_img_supervise_loss.append(loss)
    #             loss = 0
    #             batch_count = 0
    #         optimizer.zero_grad()
    #         img_supervise_batch_loss.backward()
    #         optimizer.step()
         
    #     if epoch % img_supervise_epochs == 0:
    #         filename = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
    #         torch.save(model.Imgmodel, savepath + filename + 'pretrainimgfeature.pkl')
    #         torch.save(model.Imgpredictmodel, savepath + filename + 'pretrainimgpredict.pkl')
    #         np.save(savepath + filename + "imgsuperviseloss.npy", train_img_supervise_loss)
    #         acc = Imgtest(model.Imgmodel, model.Imgpredictmodel, dataset.test_(), batchsize = imgbatchsize, cuda = cuda)
    #         print('Image supervise - acc :', acc)
    #         print()
    #         # log_image().info(f'----------Img supervise - Accuracy: {acc} --------------------------')  
    #         np.save(savepath + filename + "imgsuperviseacc.npy", [acc])
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
    # loss = 0
    # text_supervise_epochs = 1
    # print('Pretrain text supervise data:')
    # for epoch in range(1, text_supervise_epochs + 1):
    #     data_loader = DataLoader(dataset = dataset.supervise_(), batch_size = textbatchsize, shuffle = True, num_workers = 0)
    #     for batch_index, (x, y) in enumerate(data_loader, 1):
    #         batch_count += 1
    #         scheduler.step()
    #         text_xx = x[1]
    #         if args.use_bert_embedding:
    #             bert_xx = x[2]
    #         label = y
    #         text_xx = text_xx.float()
    #         if args.use_bert_embedding:
    #             bert_xx = bert_xx.float()
    #         label = label.float()
    #         text_xx = Variable(text_xx).cuda() if cuda else Variable(text_xx)  
    #         if args.use_bert_embedding:
    #             bert_xx = Variable(bert_xx).cuda() if cuda else Variable(bert_xx)  
    #         label = Variable(label).cuda() if cuda else Variable(label)  
    #         if args.use_bert_embedding:
    #             textxx = model.Textfeaturemodel(text_xx,bert_xx)
    #         else:
    #             textxx = model.Textfeaturemodel(text_xx)
    #         textyy = model.Textpredictmodel(textxx)
    #         if args.use_focal_loss:
    #             text_supervise_batch_loss = focal_binary_cross_entropy(args,textyy, label)
    #         else:
    #             text_supervise_batch_loss = criterion(textyy, label)
    #         loss += text_supervise_batch_loss.data.item()
    #         if batch_count >= loss_batch:
    #             loss = loss/loss_batch
    #             train_text_supervise_loss.append(loss)
    #             loss = 0
    #             batch_count = 0
    #         optimizer.zero_grad()
    #         text_supervise_batch_loss.backward()
    #         optimizer.step()
    #     if epoch % text_supervise_epochs == 0:
    #         filename = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
    #         torch.save(model.Textfeaturemodel, savepath + filename + 'pretraintextfeature.pkl')
    #         torch.save(model.Textpredictmodel, savepath + filename + 'pretraintextpredict.pkl')
    #         np.save(savepath + filename + "textsuperviseloss.npy", train_text_supervise_loss)
    #         acc = texttest(args,model.Textfeaturemodel,model.Textpredictmodel, dataset.test_(), batchsize = textbatchsize, cuda = cuda)
    #         print('Text supervise - acc :', acc)
    #         print()
    #         np.save(savepath + filename + "textsuperviseacc.npy", [acc])

    optimizer = optim.Adam(model.parameters(), lr = lr_supervise, weight_decay = weight_decay)
    scheduler = StepLR(optimizer, step_size = 500, gamma = 0.9)  
    criterion = torch.nn.BCELoss()

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

        for batch_index, (supbatch, unsupbatch) in tqdm(enumerate(zip(dataset['train_sup'], dataset['train_unsup'])), 
                                                total=min(len(dataset['train_sup']), len(dataset['train_unsup']))):

            (sup_img, sup_text), sup_label = supbatch
            (unsup_img, unsup_text) = unsupbatch

            sup_label = sup_label.unsqueeze(-1) if len(sup_label.shape) == 1 else sup_label # expand last dim for single label only

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
            supervise_text_xx = sup_text['sentence_vectors']
            if args.use_bert_embedding:
                supervise_bert_xx = sup_text['sbert_embedding']
            label = sup_label

            supervise_img_xx = supervise_img_xx.float()
            supervise_text_xx = supervise_text_xx.float()
            if args.use_bert_embedding:
                supervise_bert_xx = supervise_bert_xx.float()
            label = label.float()
            supervise_img_xx = Variable(supervise_img_xx).cuda() if cuda else Variable(supervise_img_xx)  
            supervise_text_xx = Variable(supervise_text_xx).cuda() if cuda else Variable(supervise_text_xx)  
            if args.use_bert_embedding:
                supervise_bert_xx = Variable(supervise_bert_xx).cuda() if cuda else Variable(supervise_bert_xx)  
            label = Variable(label).cuda() if cuda else Variable(label)  
            supervise_imghidden = model.Imgmodel(supervise_img_xx)
            if args.use_bert_embedding:
                supervise_texthidden = model.Textfeaturemodel(supervise_text_xx,supervise_bert_xx)
            else:
                supervise_texthidden = model.Textfeaturemodel(supervise_text_xx)
            if model.Projectormodel != None:
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
            if args.use_focal_loss:
                totalloss = focal_binary_cross_entropy(args,supervise_predict, label)
                imgloss = focal_binary_cross_entropy(args,supervise_imgpredict, label)
                textloss = focal_binary_cross_entropy(args,supervise_textpredict, label)
            else:
                totalloss = criterion(supervise_predict, label)
                imgloss = criterion(supervise_imgpredict, label)
                textloss = criterion(supervise_textpredict, label)
            '''
            Diversity Measure code.
            '''         
            similar = torch.bmm(supervise_imgpredict.unsqueeze(1), supervise_textpredict.unsqueeze(2)).view(supervise_imgpredict.size()[0])
            norm_matrix_img = torch.norm(supervise_imgpredict, 2, dim = 1)
            norm_matrix_text = torch.norm(supervise_textpredict, 2, dim = 1)
            div = torch.mean(similar/(norm_matrix_img * norm_matrix_text))
            
            if args.use_auto_weight:
                supervise_loss = 1/(2*model.Predictmodel.sigma[0]**2)*imgloss + 1/(2*model.Predictmodel.sigma[1]**2)*textloss \
                + 1/(2*model.Predictmodel.sigma[2]**2)*totalloss + torch.log(model.Predictmodel.sigma).sum()
            else:
                supervise_loss = imgloss + textloss + 2.0*totalloss
            epoch_img_loss_train += imgloss.item()
            epoch_text_loss_train += textloss.item() 
            epoch_total_loss_train += totalloss.item()
            '''
            Robust Consistency Measure code.
            '''

            unsupervise_img_xx = unsup_img
            unsupervise_text_xx = unsup_text['sentence_vectors']

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

            unsupervise_img_xx = unsupervise_img_xx.float()
            unsupervise_text_xx = unsupervise_text_xx.float()
            if args.use_bert_embedding:
                unsupervise_bert_xx = unsupervise_bert_xx.float()
            unsupervise_img_xx = Variable(unsupervise_img_xx).cuda() if cuda else Variable(unsupervise_img_xx)  
            unsupervise_text_xx = Variable(unsupervise_text_xx).cuda() if cuda else Variable(unsupervise_text_xx) 
            if args.use_bert_embedding:
                unsupervise_bert_xx = Variable(unsupervise_bert_xx).cuda() if cuda else Variable(unsupervise_bert_xx)    
            unsupervise_imghidden = model.Imgmodel(unsupervise_img_xx)
            if args.use_bert_embedding:
                unsupervise_texthidden = model.Textfeaturemodel(unsupervise_text_xx,unsupervise_bert_xx)
            else:
                unsupervise_texthidden = model.Textfeaturemodel(unsupervise_text_xx)

            if model.Projectormodel != None:
                vcreg_loss_unsupervise = model.Projectormodel(unsupervise_imghidden,unsupervise_texthidden)

            unsupervise_imgpredict = model.Imgpredictmodel(unsupervise_imghidden)
            unsupervise_textpredict = model.Textpredictmodel(unsupervise_texthidden)
            unsimilar = torch.bmm(unsupervise_imgpredict.unsqueeze(1), unsupervise_textpredict.unsqueeze(2)).view(unsupervise_imgpredict.size()[0])
            unnorm_matrix_img = torch.norm(unsupervise_imgpredict, 2, dim = 1)
            unnorm_matrix_text = torch.norm(unsupervise_textpredict, 2, dim = 1)
            dis = 2 - unsimilar/(unnorm_matrix_img * unnorm_matrix_text)
            tensor1 = dis[torch.abs(dis) < cita]
            tensor2 = dis[torch.abs(dis) >= cita]
            tensor1loss = torch.sum(tensor1 * tensor1/2)
            tensor2loss = torch.sum(cita * (torch.abs(tensor2) - 1/2 * cita))
            unsupervise_loss = (tensor1loss + tensor2loss)/unsupervise_img_xx.size()[0]        
            if model.Projectormodel != None:
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
            loss += total_loss.item()
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            filename = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
            torch.save(model.Imgmodel, savepath + 'supervise' + filename +'img.pkl')
            torch.save(model.Textfeaturemodel, savepath + 'supervise' + filename + 'Textfeaturemodel.pkl')
            torch.save(model.Imgpredictmodel, savepath + 'supervise' + filename + 'Imgpredictmodel.pkl')
            torch.save(model.Textpredictmodel, savepath + 'supervise' + filename + 'Textpredictmodel.pkl')
            torch.save(model.Attentionmodel, savepath + 'supervise' + filename +'attention.pkl')
        
        #===================== Multilabel =================#
        # (f1_macro_multi_1, f1_macro_multi_2, f1_macro_multi_3, total_predict, truth, f1_skl1,
        # f1_skl2, f1_skl3, f1_pm1, f1_pm2, f1_pm3,
        # auc_pm1,auc_pm2,auc_pm3, acc1, acc2, acc3, 
        # coverage1, coverage2, coverage3, example_auc1,
        # example_auc2, example_auc3, macro_auc1, macro_auc2,
        # macro_auc3, micro_auc1, micro_auc2, micro_auc3,
        # ranking_loss1, ranking_loss2, ranking_loss3,
        # humour,sarcasm,offensive,motivational,humour_truth,
        # sarcasm_truth,offensive_truth,motivational_truth) = test(args,model.Textfeaturemodel,
        # model.Imgpredictmodel, model.Textpredictmodel, model.Imgmodel,
        # model.Predictmodel, model.Attentionmodel, dataset.test_(), batchsize = batchsize, cuda = cuda)
        
        # total_step = len(data_loader)
        # epoch_supervise_loss_train = epoch_supervise_loss_train/total_step
        # epoch_div_train = epoch_div_train/total_step
        # epoch_unsupervise_loss_train = epoch_unsupervise_loss_train/total_step
        # epoch_img_loss_train = epoch_img_loss_train/total_step
        # epoch_text_loss_train = epoch_text_loss_train/total_step 
        # epoch_total_loss_train = epoch_total_loss_train/total_step

        # wandb.log({"train_loss/epoch_supervise_loss_train":epoch_supervise_loss_train})
        # wandb.log({"train_loss/epoch_div_train":epoch_div_train})
        # wandb.log({"train_loss/epoch_unsupervise_loss_train":epoch_unsupervise_loss_train})
        # wandb.log({"train_loss/epoch_img_loss_train":epoch_img_loss_train})
        # wandb.log({"train_loss/epoch_text_loss_train":epoch_text_loss_train})
        # wandb.log({"train_loss/epoch_total_loss_train":epoch_total_loss_train})

        # if model.Projectormodel != None:
        #     epoch_v_supervise_loss_train = epoch_v_supervise_loss_train/total_step
        #     epoch_c_supervise_loss_train = epoch_c_supervise_loss_train/total_step
        #     epoch_v_unsupervise_loss_train = epoch_v_unsupervise_loss_train/total_step
        #     epoch_c_unsupervise_loss_train = epoch_c_unsupervise_loss_train/total_step
        #     wandb.log({"train_loss/epoch_v_supervise_loss_train":epoch_v_supervise_loss_train})
        #     wandb.log({"train_loss/epoch_c_supervise_loss_train":epoch_c_supervise_loss_train})
        #     wandb.log({"train_loss/epoch_v_unsupervise_loss_train":epoch_v_unsupervise_loss_train})
        #     wandb.log({"train_loss/epoch_c_unsupervise_loss_train":epoch_c_unsupervise_loss_train})
        #     if args.use_sim_loss:
        #         epoch_i_supervise_loss_train = epoch_i_supervise_loss_train/total_step
        #         epoch_i_unsupervise_loss_train = epoch_i_unsupervise_loss_train/total_step
        #         wandb.log({"train_loss/epoch_i_supervise_loss_train":epoch_i_supervise_loss_train})
        #         wandb.log({"train_loss/epoch_i_unsupervise_loss_train":epoch_i_unsupervise_loss_train})

        # print("epoch_img_loss_train",epoch_img_loss_train,
        # "\t epoch_text_loss_train:",epoch_text_loss_train,
        # "\t epoch_total_loss_train:",epoch_total_loss_train)
        
        # print("epoch_supervise_loss_train:",epoch_supervise_loss_train,
        # '\t epoch_div_train:',epoch_div_train,'\t epoch_unsupervise_loss_train',epoch_unsupervise_loss_train)
        # if model.Projectormodel != None:
        #     print("epoch_v_supervise_loss_train:",epoch_v_supervise_loss_train,
        #     "\t epoch_c_supervise_loss_train:",epoch_c_supervise_loss_train)

        #     print("epoch_v_unsupervise_loss_train:",epoch_v_unsupervise_loss_train,
        #     "\t epoch_c_unsupervise_loss_train:",epoch_c_unsupervise_loss_train)
        #     if args.use_sim_loss:
        #         print("epoch_i_supervise_loss_train:",epoch_i_supervise_loss_train,
        #         "\t epoch_i_unsupervise_loss_train:",epoch_i_unsupervise_loss_train)

        # wandb.log({"learning rate/lr":scheduler.get_last_lr()[0]})
        # wandb.log({"f1_macro_multi_1":f1_macro_multi_1})
        # wandb.log({"f1_macro_multi_2":f1_macro_multi_2})
        # wandb.log({"f1_macro_multi_3":f1_macro_multi_3})
        
        # wandb.log({"f1_skl_all":f1_skl1})
        # wandb.log({"f1_skl_image":f1_skl2})
        # wandb.log({"f1_skl_text":f1_skl3})

        # wandb.log({"f1_pytorch_all":f1_pm1})
        # wandb.log({"f1_pytorch_image":f1_pm2})
        # wandb.log({"f1_pytorch_text":f1_pm3})

        # wandb.log({"Prediction": total_predict })
        # wandb.log({"Ground_truth": truth })


        # # wandb.log({"roc": wandb.plot.roc_curve(truth[:,0], np.expand_dims(total_predict[:,0], axis=-1 ))})
        # # wandb.log({"roc": wandb.plot.roc_curve(truth[:,1], np.expand_dims(total_predict[:,1], axis=-1 ) )})
        # # wandb.log({"roc": wandb.plot.roc_curve(truth[:,2], np.expand_dims(total_predict[:,2], axis=-1 ) )})
        # # wandb.log({"roc": wandb.plot.roc_curve(truth[:,3], np.expand_dims(total_predict[:,3], axis=-1 ) )})

        # wandb.log({f"roc/epoch{epoch}/_roc_humour": wandb.plot.roc_curve(truth[:,0], np.stack([ 1-total_predict[:,0], total_predict[:,0] ], axis=1)  ) })
        # wandb.log({f"roc/epoch{epoch}/_roc_sarcasm": wandb.plot.roc_curve( truth[:,1], np.stack([ 1-total_predict[:,1], total_predict[:,1] ], axis=1) ) })
        # wandb.log({f"roc/epoch{epoch}/_roc_offensive": wandb.plot.roc_curve(truth[:,2], np.stack([ 1-total_predict[:,2], total_predict[:,2] ], axis=1) ) })
        # wandb.log({f"roc/epoch{epoch}/_roc_motivational": wandb.plot.roc_curve(truth[:,3], np.stack([ 1-total_predict[:,3], total_predict[:,3] ], axis=1) ) })

        # total_2 = total_predict > 0.5

        # # wandb.log({"conf_mat": wandb.plot.confusion_matrix( truth[:,0],total_2[:,0]) })
        # # wandb.log({"conf_mat": wandb.plot.confusion_matrix( truth[:,1],total_2[:,1]) })
        # # wandb.log({"conf_mat": wandb.plot.confusion_matrix( truth[:,2],total_2[:,2]) })
        # # wandb.log({"conf_mat": wandb.plot.confusion_matrix( truth[:,3],total_2[:,3]) })

        # # total_2 = a > 0.5

        # wandb.log({f"conf_mat/epoch{epoch}/_confmat_humour": wandb.plot.confusion_matrix( y_true = truth[:,0], preds = total_2[:,0], class_names=["not humour", "humour"] )})
        # wandb.log({f"conf_mat/epoch{epoch}/_confmat_sarcasm": wandb.plot.confusion_matrix( y_true = truth[:,1], preds = total_2[:,1], class_names=["not sarcasm", "sarcasm"] )})
        # wandb.log({f"conf_mat/epoch{epoch}/_confmat_offensive": wandb.plot.confusion_matrix( y_true = truth[:,2], preds = total_2[:,2], class_names=["not offensive", "offensive"] )})
        # wandb.log({f"conf_mat/epoch{epoch}/_confmat_motivational": wandb.plot.confusion_matrix( y_true = truth[:,3], preds = total_2[:,3], class_names=["not motivational", "motivational"] )})

        # # , labels = ["humour"]
        # # , labels = ["sarcasm"]
        # # , labels = ["offensive"]
        # # , labels = ["motivational"]

        # wandb.log({"histogram/_hist_label_humour_pred":wandb.Histogram(np_histogram = humour)})
        # wandb.log({"histogram/_hist_label_sarcasm_pred":wandb.Histogram(np_histogram = sarcasm)})
        # wandb.log({"histogram/_hist_label_offensive_pred":wandb.Histogram(np_histogram = offensive)})
        # wandb.log({"histogram/_hist_label_motivational_pred":wandb.Histogram(np_histogram = motivational)})

        # wandb.log({"histogram/_hist_label_humour_truth":wandb.Histogram(np_histogram = humour_truth)})
        # wandb.log({"histogram/_hist_label_sarcasm_truth":wandb.Histogram(np_histogram = sarcasm_truth)})
        # wandb.log({"histogram/_hist_label_offensive_truth":wandb.Histogram(np_histogram = offensive_truth)})
        # wandb.log({"histogram/_hist_label_motivational_truth":wandb.Histogram(np_histogram = motivational_truth)})


        # print('f1_skl:    ', f1_skl1,'\t', f1_skl2,'\t', f1_skl3)
        # print('f1_pm:    ', f1_pm1,'\t', f1_pm2,'\t', f1_pm3)
        # # print('coverage:    ', coverage1,'\t', coverage2,'\t', coverage3)
        # # print('auc_pm:    ', auc_pm1,'\t', auc_pm2,'\t', auc_pm3)
        # # print('example_auc: ',  example_auc1,'\t', example_auc2,'\t', example_auc3)
        # # print('macro_auc:   ',  macro_auc1,'\t', macro_auc2,'\t', macro_auc3)
        # # print('micro_auc:   ',  micro_auc1,'\t', micro_auc2,'\t', micro_auc3)
        # # print('ranking_loss:',  ranking_loss1,'\t', ranking_loss2,'\t', ranking_loss3)
        # print()

        #=================== Single label =====================#
        macro_f1_all,macro_f1_image,macro_f1_text, macro_roc_auc1, macro_roc_auc2, macro_roc_auc3, total_predict, truth, misogynous, misogynous_truth = test_singlelabel(args, model.Textfeaturemodel, model.Imgpredictmodel, model.Textpredictmodel, model.Imgmodel, model.Predictmodel, model.Attentionmodel, dataset['val'], batchsize = batchsize, cuda = cuda)
        
        #print("------Epoch : ", epoch, "------" )
        #log_train().info(f'\n --------Epoch :{epoch}--------------')
        #print("-------------- Total ---------------------- Image -------------------- Text----")
        #log_train().info(f'-------------- Total ---------------------- Image -------------------- Text----')
        # print('Acc:         ', acc1,'\t', acc2,'\t', acc3)
        # log_train().info(f'\n[logger] Acc = \n{acc1},{acc2},{acc3}'
        #                                 f'\n{"-" * 100}')
        # log_train().info(f'\n Acc:      {acc1}      {acc2}    {acc3}')
        # log_train().info(f'\n coverage:      {coverage1}      {coverage2}    {coverage3}')
        # log_train().info(f'\n example_auc:      {example_auc1}      {example_auc2}    {example_auc3}')
        # log_train().info(f'\n macro_auc:      {macro_auc1}      {macro_auc2}    {macro_auc3}')
        # log_train().info(f'\n micro_auc:      {micro_auc1}      {micro_auc2}    {micro_auc3}')
        # log_train().info(f'\n ranking_loss:      {ranking_loss1}      {ranking_loss2}    {ranking_loss3}')
        # log_train().info(f'\n --------Loss :{loss}--------------')

            
            # wandb.log({"f1_skl_all":f1_skl1})
            # wandb.log({"f1_skl_image":f1_skl2})
            # wandb.log({"f1_skl_text":f1_skl3})

        wandb.log({"macro_roc_auc_single":macro_roc_auc1})
        wandb.log({"macro_roc_auc_image_single":macro_roc_auc2})
        wandb.log({"macro_roc_auc_text_single":macro_roc_auc3})

            # wandb.log({"Prediction": total_predict })
            # wandb.log({"Ground_truth": truth })
        

        wandb.log({f"roc/epoch{epoch}/_roc_misogynous": wandb.plot.roc_curve(truth[:,0], np.stack([ 1-total_predict[:,0], total_predict[:,0] ], axis=1)  ) })


        total_2 = total_predict > 0.5
        wandb.log({f"conf_mat/epoch{epoch}/_confmat_misogynous": wandb.plot.confusion_matrix( y_true = truth[:,0], preds = total_2[:,0], class_names=["not misogynous", "misogynous"] )})


            # wandb.log({"histogram/_hist_label_humour_pred":wandb.Histogram(np_histogram = humour)})
            # wandb.log({"histogram/_hist_label_sarcasm_pred":wandb.Histogram(np_histogram = sarcasm)})
            # wandb.log({"histogram/_hist_label_offensive_pred":wandb.Histogram(np_histogram = offensive)})
            # wandb.log({"histogram/_hist_label_motivational_pred":wandb.Histogram(np_histogram = motivational)})

            # wandb.log({"histogram/_hist_label_humour_truth":wandb.Histogram(np_histogram = humour_truth)})
            # wandb.log({"histogram/_hist_label_sarcasm_truth":wandb.Histogram(np_histogram = sarcasm_truth)})
            # wandb.log({"histogram/_hist_label_offensive_truth":wandb.Histogram(np_histogram = offensive_truth)})
            # wandb.log({"histogram/_hist_label_motivational_truth":wandb.Histogram(np_histogram = motivational_truth)})

        wandb.log({"macro_f1_all":macro_f1_all})
        wandb.log({"macro_f1_image":macro_f1_image})
        wandb.log({"macro_f1_text":macro_f1_text})

        print('f1_macro:    ', macro_f1_all,'\t', macro_f1_image,'\t', macro_f1_text)
        # print('f1_pm:    ', f1_pm1,'\t', f1_pm2,'\t', f1_pm3)
        # print('coverage:    ', coverage1,'\t', coverage2,'\t', coverage3)
        # print('auc_pm:    ', auc_pm1,'\t', auc_pm2,'\t', auc_pm3)
        # print('example_auc: ',  example_auc1,'\t', example_auc2,'\t', example_auc3)
        # print('macro_auc:   ',  macro_auc1,'\t', macro_auc2,'\t', macro_auc3)
        # print('micro_auc:   ',  micro_auc1,'\t', micro_auc2,'\t', micro_auc3)
        # print('ranking_loss:',  ranking_loss1,'\t', ranking_loss2,'\t', ranking_loss3)
        print()
        #np.save(savepath + filename + "superviseacc.npy", [acc1, acc2, acc3])

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

    wandb.run.name = args.experiment

    if args.use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpu
        cuda = torch.cuda.is_available() and args.use_gpu

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

    train_supervised_loader, train_unsupervised_loader, val_loader = create_semi_supervised_dataloaders(args, 
                                            train_img_dir='data/MAMI_processed/images/train',
                                            train_labeled_csv='data/MAMI_processed/train_labeled_ratio-0.3.csv',
                                            train_unlabeled_csv='data/MAMI_processed/train_unlabeled_ratio-0.3.csv',
                                            val_img_dir = 'data/MAMI_processed/images/val',
                                            val_csv='data/MAMI_processed/val.csv',
                                            batch_size=args.batchsize, image_size=256)

    dataset = {'train_sup': train_supervised_loader,
                'train_unsup': train_unsupervised_loader,
                'val': val_loader}

    model = CmmlModel(args).cuda()

    savepath_folder = args.savepath+"/"+args.experiment 
    if not os.path.exists(args.savepath):
        os.mkdir(args.savepath)
    if not os.path.exists(savepath_folder):
        os.mkdir(savepath_folder)

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



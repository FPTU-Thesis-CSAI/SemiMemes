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
from evaluation_metric.measure_average_precision import * 
from evaluation_metric.f1_metric import *
from evaluation_metric.measure_coverage import * 
from evaluation_metric.measure_example_auc import *
from evaluation_metric.measure_macro_auc import * 
from evaluation_metric.measure_ranking_loss import * 
from evaluation_metric.measure_micro_auc import * 
from evaluation_metric.roc_auc import *
import torch.nn as nn
import wandb   
from tqdm import tqdm

def test(Textfeaturemodel, Imgpredictmodel, Textpredictmodel, Imgmodel, Predictmodel, Attentionmodel, testdataset, batchsize = 32, cuda = False):
    if cuda:
        Textfeaturemodel.cuda()
        Imgpredictmodel.cuda()
        Textpredictmodel.cuda()
        Imgmodel.cuda()
        Predictmodel.cuda()
        Attentionmodel.cuda()
    Textfeaturemodel.eval()
    Imgpredictmodel.eval()
    Textpredictmodel.eval()
    Imgmodel.eval()
    Predictmodel.eval()
    Attentionmodel.eval()
    print('----------------- Test data:------------')
    data_loader = DataLoader(dataset = testdataset, batch_size = batchsize, shuffle = False)
    total_predict = []
    img_predict = []
    text_predict = []
    truth = []
    for batch_index, (x, y) in enumerate(data_loader, 1):
        img_xx = x[0]
        text_xx = x[1]
        bert_xx = x[2]
        label = y.numpy()
        img_xx = img_xx.float()
        text_xx = text_xx.float()
        bert_xx = bert_xx.float()
        img_xx = Variable(img_xx).cuda() if cuda else Variable(img_xx)
        text_xx = Variable(text_xx).cuda() if cuda else Variable(text_xx)
        bert_xx = Variable(bert_xx).cuda() if cuda else Variable(bert_xx)
        imghidden = Imgmodel(img_xx)
        texthidden = Textfeaturemodel(text_xx,bert_xx)

        imgk = Attentionmodel(imghidden)
        textk = Attentionmodel(texthidden)
        modality_attention = []
        modality_attention.append(imgk)
        modality_attention.append(textk)
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
        imgpredict = Imgpredictmodel(imghidden)
        textpredict = Textpredictmodel(texthidden)
        feature_hidden = img_attention * imghidden + text_attention * texthidden
        predict = Predictmodel(feature_hidden)
        img_ = imgpredict.cpu().data.numpy()
        text_ = textpredict.cpu().data.numpy()
        predict = predict.cpu().data.numpy()
        total_predict.append(predict)
        img_predict.append(img_)
        text_predict.append(text_)
        truth.append(label)

    total_predict = np.array(total_predict)
    img_predict = np.array(img_predict)
    text_predict = np.array(text_predict)
    truth = np.array(truth)
    temp = total_predict[0]
    for i in range(1, len(total_predict)):
        temp = np.vstack((temp, total_predict[i]))
    total_predict = temp
    temp = img_predict[0]
    for i in range(1, len(img_predict)):
        temp = np.vstack((temp, img_predict[i]))
    img_predict = temp
    temp = text_predict[0]
    for i in range(1, len(text_predict)):
        temp = np.vstack((temp, text_predict[i]))
    text_predict = temp
    temp = truth[0]
    for i in range(1, len(truth)):
        temp = np.vstack((temp, truth[i]))
    truth = temp

    f1_macro_multi_1 = macro_f1_multilabel(total_predict, truth, num_labels=4, threshold = 0.5, reduce = True)
    f1_macro_multi_2 = macro_f1_multilabel(img_predict, truth, num_labels=4, threshold = 0.5, reduce = True)
    f1_macro_multi_3 = macro_f1_multilabel(text_predict, truth,  num_labels=4, threshold = 0.5, reduce = True)

    f1_skl1 = f1_score_sklearn(total_predict, truth)
    f1_skl2 = f1_score_sklearn(img_predict, truth)
    f1_skl3 = f1_score_sklearn(text_predict, truth)

    f1_pm1 = f1_score_pytorch(total_predict, truth)
    f1_pm2 = f1_score_pytorch(img_predict, truth)
    f1_pm3 = f1_score_pytorch(text_predict, truth)

    auc_pm1 = auroc_score_pytorch(total_predict, truth)
    auc_pm2 = auroc_score_pytorch(img_predict, truth)
    auc_pm3 = auroc_score_pytorch(text_predict, truth)

    average_precison1 = average_precision(total_predict, truth)
    average_precison2 = average_precision(img_predict, truth)
    average_precison3 = average_precision(text_predict, truth)
    
    coverage1 = coverage(total_predict, truth)
    coverage2 = coverage(img_predict, truth)
    coverage3 = coverage(text_predict, truth)
    
    example_auc1 = example_auc(total_predict, truth)
    example_auc2 = example_auc(img_predict, truth)
    example_auc3 = example_auc(text_predict, truth)

    macro_auc1 = macro_auc(total_predict, truth)
    macro_auc2 = macro_auc(img_predict, truth)
    macro_auc3 = macro_auc(text_predict, truth)

    micro_auc1 = micro_auc(total_predict, truth)
    micro_auc2 = micro_auc(img_predict, truth)
    micro_auc3 = micro_auc(text_predict, truth)

    ranking_loss1 = ranking_loss(total_predict, truth)
    ranking_loss2 = ranking_loss(img_predict, truth)
    ranking_loss3 = ranking_loss(text_predict, truth)
    
    humour = np.histogram(total_predict[:,0])
    sarcasm = np.histogram(total_predict[:,1])
    offensive = np.histogram(total_predict[:,2])
    motivational = np.histogram(total_predict[:,3])

    humour_truth = np.histogram(truth[:,0])
    sarcasm_truth = np.histogram(truth[:,1])
    offensive_truth = np.histogram(truth[:,2])
    motivational_truth = np.histogram(truth[:,3])

    return (f1_macro_multi_1, f1_macro_multi_2, f1_macro_multi_3, 
    total_predict, truth, f1_skl1, f1_skl2, f1_skl3, f1_pm1, f1_pm2, 
    f1_pm3, auc_pm1, auc_pm2, auc_pm3, average_precison1, average_precison2, 
    average_precison3, coverage1, coverage2, coverage3, example_auc1, 
    example_auc2, example_auc3, macro_auc1, macro_auc2, macro_auc3, 
    micro_auc1, micro_auc2, micro_auc3, ranking_loss1, ranking_loss2, 
    ranking_loss3, humour,sarcasm,offensive,motivational,humour_truth,
    sarcasm_truth,offensive_truth,motivational_truth)

def texttest(Textfeaturemodel, Textpredictmodel, testdataset, batchsize = 32, cuda = False):
    if cuda:
        Textfeaturemodel.cuda()
        Textpredictmodel.cuda()
    Textfeaturemodel.eval()
    Textpredictmodel.eval()

    print('-----------------Test Text data:---------------------')
    data_loader = DataLoader(dataset = testdataset, batch_size = batchsize, shuffle = False, num_workers = 0)
    text_predict = []
    truth = []
    for batch_index, (x, y) in enumerate(data_loader, 1):
        text_xx = x[1]
        bert_xx = x[2]
        label = y.numpy()
        text_xx = text_xx.float()
        bert_xx = bert_xx.float()
        text_xx = Variable(text_xx).cuda() if cuda else Variable(text_xx)
        bert_xx = Variable(bert_xx).cuda() if cuda else Variable(bert_xx)
        textxx = Textfeaturemodel(text_xx,bert_xx)
        textyy = Textpredictmodel(textxx)
        text_ = textyy.cpu().data.numpy()
        text_predict.append(text_)
        truth.append(label)
    text_predict = np.array(text_predict)
    truth = np.array(truth)
    temp = text_predict[0]
    for i in range(1, len(text_predict)):
        temp = np.vstack((temp, text_predict[i]))
    text_predict = temp
    temp = truth[0]
    for i in range(1, len(truth)):
        temp = np.vstack((temp, truth[i]))
    truth = temp
    average_precison = average_precision(text_predict, truth)
    return average_precison

def Imgtest(Imgmodel, Imgpredictmodel, testdataset, batchsize = 32, cuda = False):
    if cuda:
        Imgmodel.cuda()
        Imgpredictmodel.cuda()
    Imgmodel.eval()
    Imgpredictmodel.eval()
    print(f'------------------Test IMG data:-------------------')
    data_loader = DataLoader(dataset = testdataset, batch_size = batchsize, shuffle = False, num_workers = 0)
    img_predict = []
    truth = []
    for batch_index, (x, y) in enumerate(data_loader, 1):
        img_xx = x[0]
        label = y.numpy()
        img_xx = img_xx.float()
        img_xx = Variable(img_xx).cuda() if cuda else Variable(img_xx)
        imgxx = Imgmodel(img_xx)
        imgyy = Imgpredictmodel(imgxx)
        img_ = imgyy.cpu().data.numpy()
        img_predict.append(img_)
        truth.append(label)
   
    img_predict = np.array(img_predict)
    truth = np.array(truth)
    temp = img_predict[0]
    for i in range(1, len(img_predict)):
        temp = np.vstack((temp, img_predict[i]))
    img_predict = temp
    temp = truth[0]
    for i in range(1, len(truth)):
        temp = np.vstack((temp, truth[i]))
    truth = temp
    average_precison = average_precision(img_predict, truth)
    return average_precison


def train(model, dataset,
          supervise_epochs = 200, text_supervise_epochs = 50, img_supervise_epochs = 50, 
          lr_supervise = 0.01, text_lr_supervise = 0.0001, img_lr_supervise = 0.0001,
          weight_decay = 0, batchsize = 32,lambda1=0.01,lambda2=1, textbatchsize = 32,
           imgbatchsize = 32, cuda = False, savepath = ''): 
    model.train()
    print("train")
    par = []
    par.append({'params': model.Imgmodel.parameters()})
    par.append({'params': model.Imgpredictmodel.parameters()})
    optimizer = optim.Adam(par, lr = img_lr_supervise, weight_decay = weight_decay)
    scheduler = StepLR(optimizer, step_size = 500, gamma = 0.9) 
    criterion = torch.nn.BCELoss()
    train_img_supervise_loss = []
    batch_count = 0
    loss = 0
    cita = 1.003
    loss_batch = 50
    print("Pretrain img supervise data :")  
    for epoch in range(1, img_supervise_epochs + 1):
        data_loader = DataLoader(dataset = dataset.supervise_(), batch_size = imgbatchsize, shuffle = True, num_workers = 0)
        for batch_index, (x, y) in enumerate(data_loader, 1):
            batch_count += 1
            scheduler.step()
            img_xx = x[0]
            label = y
            img_xx = img_xx.float()
            label = label.float()
            img_xx = Variable(img_xx).cuda() if cuda else Variable(img_xx)  
            label = Variable(label).cuda() if cuda else Variable(label)  
            imgxx = model.Imgmodel(img_xx)
            imgyy = model.Imgpredictmodel(imgxx)
            img_supervise_batch_loss = criterion(imgyy, label)
            loss += img_supervise_batch_loss.data.item()
            if batch_count >= loss_batch:
                loss = loss/loss_batch
                train_img_supervise_loss.append(loss)
                loss = 0
                batch_count = 0
            optimizer.zero_grad()
            img_supervise_batch_loss.backward()
            optimizer.step()
         
        if epoch % img_supervise_epochs == 0:
            filename = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
            torch.save(model.Imgmodel, savepath + filename + 'pretrainimgfeature.pkl')
            torch.save(model.Imgpredictmodel, savepath + filename + 'pretrainimgpredict.pkl')
            np.save(savepath + filename + "imgsuperviseloss.npy", train_img_supervise_loss)
            acc = Imgtest(model.Imgmodel, model.Imgpredictmodel, dataset.test_(), batchsize = imgbatchsize, cuda = cuda)
            print('Image supervise - acc :', acc)
            print()
            # log_image().info(f'----------Img supervise - Accuracy: {acc} --------------------------')  
            np.save(savepath + filename + "imgsuperviseacc.npy", [acc])
    '''
    pretrain TextNet.
    ''' 
    par = []
    par.append({'params': model.Textfeaturemodel.parameters()})
    par.append({'params': model.Textpredictmodel.parameters()})
    optimizer = optim.Adam(par, lr = text_lr_supervise, weight_decay = weight_decay)
    scheduler = StepLR(optimizer, step_size = 500, gamma = 0.9) 
    criterion = torch.nn.BCELoss()
    train_text_supervise_loss = []
    batch_count = 0
    loss = 0
    text_supervise_epochs = 1
    print('Pretrain text supervise data:')
    for epoch in range(1, text_supervise_epochs + 1):
        data_loader = DataLoader(dataset = dataset.supervise_(), batch_size = textbatchsize, shuffle = True, num_workers = 0)
        for batch_index, (x, y) in enumerate(data_loader, 1):
            batch_count += 1
            scheduler.step()
            text_xx = x[1]
            bert_xx = x[2]
            label = y
            text_xx = text_xx.float()
            bert_xx = bert_xx.float()
            label = label.float()
            text_xx = Variable(text_xx).cuda() if cuda else Variable(text_xx)  
            bert_xx = Variable(bert_xx).cuda() if cuda else Variable(bert_xx)  
            label = Variable(label).cuda() if cuda else Variable(label)  
            textxx = model.Textfeaturemodel(text_xx,bert_xx)
            textyy = model.Textpredictmodel(textxx)
            text_supervise_batch_loss = criterion(textyy, label)
            loss += text_supervise_batch_loss.data.item()
            if batch_count >= loss_batch:
                loss = loss/loss_batch
                train_text_supervise_loss.append(loss)
                loss = 0
                batch_count = 0
            optimizer.zero_grad()
            text_supervise_batch_loss.backward()
            optimizer.step()
        if epoch % text_supervise_epochs == 0:
            filename = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
            torch.save(model.Textfeaturemodel, savepath + filename + 'pretraintextfeature.pkl')
            torch.save(model.Textpredictmodel, savepath + filename + 'pretraintextpredict.pkl')
            np.save(savepath + filename + "textsuperviseloss.npy", train_text_supervise_loss)
            acc = texttest(model.Textfeaturemodel,model.Textpredictmodel, dataset.test_(), batchsize = textbatchsize, cuda = cuda)
            print('Text supervise - acc :', acc)
            print()
            np.save(savepath + filename + "textsuperviseacc.npy", [acc])

    optimizer = optim.Adam(model.parameters(), lr = lr_supervise, weight_decay = weight_decay)
    scheduler = StepLR(optimizer, step_size = 500, gamma = 0.9)  
    criterion = torch.nn.BCELoss()

    train_supervise_loss = []
    batch_count = 0
    loss = 0
    for epoch in range(1, supervise_epochs + 1):
        print('train supervise data:', epoch)
        data_loader = DataLoader(dataset = dataset.unsupervise_(), batch_size = batchsize, shuffle = True, num_workers = 0)
        for batch_index, (x, y) in tqdm(enumerate(data_loader, 1)):
            batch_count += 1
            scheduler.step()
            x[0] = torch.cat(x[0], 0)
            x[1] = torch.cat(x[1], 0)
            x[2] = torch.cat(x[2], 0)
            y = torch.cat(y, 0)
            '''
            Attention architecture and use bceloss.
            '''
            supervise_img_xx = x[0]
            supervise_text_xx = x[1]
            supervise_bert_xx = x[2]
            label = y
            supervise_img_xx = supervise_img_xx.float()
            supervise_text_xx = supervise_text_xx.float()
            supervise_bert_xx = supervise_bert_xx.float()
            label = label.float()
            supervise_img_xx = Variable(supervise_img_xx).cuda() if cuda else Variable(supervise_img_xx)  
            supervise_text_xx = Variable(supervise_text_xx).cuda() if cuda else Variable(supervise_text_xx)  
            supervise_bert_xx = Variable(supervise_bert_xx).cuda() if cuda else Variable(supervise_bert_xx)  
            label = Variable(label).cuda() if cuda else Variable(label)  
            supervise_imghidden = model.Imgmodel(supervise_img_xx)
            supervise_texthidden = model.Textfeaturemodel(supervise_text_xx,supervise_bert_xx)
            vcreg_loss = model.Projectormodel(supervise_imghidden,supervise_texthidden)
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
    
            supervise_loss = 1/(2*model.Predictmodel.sigma[0]**2)*imgloss + 1/(2*model.Predictmodel.sigma[1]**2)*textloss \
            + 1/(2*model.Predictmodel.sigma[2]**2)*totalloss + torch.log(model.Predictmodel.sigma).sum()
            '''
            Robust Consistency Measure code.
            '''
            x[3] = torch.cat(x[3], 0)
            x[4] = torch.cat(x[4], 0)
            x[5] = torch.cat(x[5], 0)
            unsupervise_img_xx = x[3]
            unsupervise_text_xx = x[4]
            unsupervise_bert_xx = x[5]
            unsupervise_img_xx = unsupervise_img_xx.float()
            unsupervise_text_xx = unsupervise_text_xx.float()
            unsupervise_bert_xx = unsupervise_bert_xx.float()
            unsupervise_img_xx = Variable(unsupervise_img_xx).cuda() if cuda else Variable(unsupervise_img_xx)  
            unsupervise_text_xx = Variable(unsupervise_text_xx).cuda() if cuda else Variable(unsupervise_text_xx) 
            unsupervise_bert_xx = Variable(unsupervise_bert_xx).cuda() if cuda else Variable(unsupervise_bert_xx)    
            unsupervise_imghidden = model.Imgmodel(unsupervise_img_xx)
            unsupervise_texthidden = model.Textfeaturemodel(unsupervise_text_xx,unsupervise_bert_xx)
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
            total_loss = supervise_loss + 0.01* div +  unsupervise_loss + vcreg_loss
            
            loss += total_loss.data.item()
            if batch_count >= loss_batch:
                loss = loss/loss_batch
                train_supervise_loss.append(loss)
                loss = 0
                batch_count = 0
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        if epoch % 1 == 0:
            torch.save(model.Imgmodel, savepath + 'supervise' + filename +'img.pkl')
            torch.save(model.Textfeaturemodel, savepath + 'supervise' + filename + 'Textfeaturemodel.pkl')
            torch.save(model.Imgpredictmodel, savepath + 'supervise' + filename + 'Imgpredictmodel.pkl')
            torch.save(model.Textpredictmodel, savepath + 'supervise' + filename + 'Textpredictmodel.pkl')
            torch.save(model.Attentionmodel, savepath + 'supervise' + filename +'attention.pkl')
            
            (f1_macro_multi_1, f1_macro_multi_2, f1_macro_multi_3, total_predict, truth, f1_skl1,
            f1_skl2, f1_skl3, f1_pm1, f1_pm2, f1_pm3,
            auc_pm1,auc_pm2,auc_pm3, acc1, acc2, acc3, 
            coverage1, coverage2, coverage3, example_auc1,
            example_auc2, example_auc3, macro_auc1, macro_auc2,
            macro_auc3, micro_auc1, micro_auc2, micro_auc3,
            ranking_loss1, ranking_loss2, ranking_loss3,
            humour,sarcasm,offensive,motivational,humour_truth,
            sarcasm_truth,offensive_truth,motivational_truth) = test(model.Textfeaturemodel,
            model.Imgpredictmodel, model.Textpredictmodel, model.Imgmodel,
            model.Predictmodel, model.Attentionmodel, dataset.test_(), batchsize = batchsize, cuda = cuda)
            
            wandb.log({"learning rate/lr":scheduler.get_last_lr()[0]})
            wandb.log({"f1_macro_multi_1":f1_macro_multi_1})
            wandb.log({"f1_macro_multi_2":f1_macro_multi_2})
            wandb.log({"f1_macro_multi_3":f1_macro_multi_3})
            
            wandb.log({"f1_skl_all":f1_skl1})
            wandb.log({"f1_skl_image":f1_skl2})
            wandb.log({"f1_skl_text":f1_skl3})

            wandb.log({"f1_pytorch_all":f1_pm1})
            wandb.log({"f1_pytorch_image":f1_pm2})
            wandb.log({"f1_pytorch_text":f1_pm3})

            wandb.log({"Prediction": total_predict })
            wandb.log({"Ground_truth": truth })


            # wandb.log({"roc": wandb.plot.roc_curve(truth[:,0], np.expand_dims(total_predict[:,0], axis=-1 ))})
            # wandb.log({"roc": wandb.plot.roc_curve(truth[:,1], np.expand_dims(total_predict[:,1], axis=-1 ) )})
            # wandb.log({"roc": wandb.plot.roc_curve(truth[:,2], np.expand_dims(total_predict[:,2], axis=-1 ) )})
            # wandb.log({"roc": wandb.plot.roc_curve(truth[:,3], np.expand_dims(total_predict[:,3], axis=-1 ) )})

            wandb.log({f"roc/epoch{epoch}/_roc_humour": wandb.plot.roc_curve(truth[:,0], np.stack([ 1-total_predict[:,0], total_predict[:,0] ], axis=1)  ) })
            wandb.log({f"roc/epoch{epoch}/_roc_sarcasm": wandb.plot.roc_curve( truth[:,1], np.stack([ 1-total_predict[:,1], total_predict[:,1] ], axis=1) ) })
            wandb.log({f"roc/epoch{epoch}/_roc_offensive": wandb.plot.roc_curve(truth[:,2], np.stack([ 1-total_predict[:,2], total_predict[:,2] ], axis=1) ) })
            wandb.log({f"roc/epoch{epoch}/_roc_motivational": wandb.plot.roc_curve(truth[:,3], np.stack([ 1-total_predict[:,3], total_predict[:,3] ], axis=1) ) })

            total_2 = total_predict > 0.5

            # wandb.log({"conf_mat": wandb.plot.confusion_matrix( truth[:,0],total_2[:,0]) })
            # wandb.log({"conf_mat": wandb.plot.confusion_matrix( truth[:,1],total_2[:,1]) })
            # wandb.log({"conf_mat": wandb.plot.confusion_matrix( truth[:,2],total_2[:,2]) })
            # wandb.log({"conf_mat": wandb.plot.confusion_matrix( truth[:,3],total_2[:,3]) })

            # total_2 = a > 0.5

            wandb.log({f"conf_mat/epoch{epoch}/_confmat_humour": wandb.plot.confusion_matrix( y_true = truth[:,0], preds = total_2[:,0], class_names=["not humour", "humour"] )})
            wandb.log({f"conf_mat/epoch{epoch}/_confmat_sarcasm": wandb.plot.confusion_matrix( y_true = truth[:,1], preds = total_2[:,1], class_names=["not sarcasm", "sarcasm"] )})
            wandb.log({f"conf_mat/epoch{epoch}/_confmat_offensive": wandb.plot.confusion_matrix( y_true = truth[:,2], preds = total_2[:,2], class_names=["not offensive", "offensive"] )})
            wandb.log({f"conf_mat/epoch{epoch}/_confmat_motivational": wandb.plot.confusion_matrix( y_true = truth[:,3], preds = total_2[:,3], class_names=["not motivational", "motivational"] )})

            # , labels = ["humour"]
            # , labels = ["sarcasm"]
            # , labels = ["offensive"]
            # , labels = ["motivational"]

            wandb.log({"histogram/_hist_label_humour_pred":wandb.Histogram(np_histogram = humour)})
            wandb.log({"histogram/_hist_label_sarcasm_pred":wandb.Histogram(np_histogram = sarcasm)})
            wandb.log({"histogram/_hist_label_offensive_pred":wandb.Histogram(np_histogram = offensive)})
            wandb.log({"histogram/_hist_label_motivational_pred":wandb.Histogram(np_histogram = motivational)})

            wandb.log({"histogram/_hist_label_humour_truth":wandb.Histogram(np_histogram = humour_truth)})
            wandb.log({"histogram/_hist_label_sarcasm_truth":wandb.Histogram(np_histogram = sarcasm_truth)})
            wandb.log({"histogram/_hist_label_offensive_truth":wandb.Histogram(np_histogram = offensive_truth)})
            wandb.log({"histogram/_hist_label_motivational_truth":wandb.Histogram(np_histogram = motivational_truth)})


            print('f1_skl:    ', f1_skl1,'\t', f1_skl2,'\t', f1_skl3)
            print('f1_pm:    ', f1_pm1,'\t', f1_pm2,'\t', f1_pm3)
            # print('coverage:    ', coverage1,'\t', coverage2,'\t', coverage3)
            # print('auc_pm:    ', auc_pm1,'\t', auc_pm2,'\t', auc_pm3)
            # print('example_auc: ',  example_auc1,'\t', example_auc2,'\t', example_auc3)
            # print('macro_auc:   ',  macro_auc1,'\t', macro_auc2,'\t', macro_auc3)
            # print('micro_auc:   ',  micro_auc1,'\t', micro_auc2,'\t', micro_auc3)
            # print('ranking_loss:',  ranking_loss1,'\t', ranking_loss2,'\t', ranking_loss3)
            print()
    return 

if __name__ == '__main__':
    args = get_args()
    wandb.init(project="meme_experiments", entity="meme-analysts",mode="disabled")
    wandb.run.name = args.experiment
    if args.use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpu
        cuda = torch.cuda.is_available() and args.use_gpu

    dataset = MemotionDatasetForCmml(args.imgfilenamerecord, 
                        args.imgfilenamerecord_unlabel, 
                        args.imgfilename, args.textfilename, 
                        args.textfilename_unlabel, 
                        args.labelfilename, 
                        args.labelfilename_unlabel, 
                        args.imgfilenamerecord_val, 
                        args.imgfilename_val, 
                        args.textfilename_val, 
                        args.labelfilename_val,
                        args.sbertemb,
                        args.sbertemb_unlabel,
                        args.sbertemb_val,
                        train = True, 
                        supervise = True)

    model = CmmlModel(args).cuda()

    savepath_folder = args.savepath+"/"+args.experiment 
    if not os.path.exists(args.savepath):
        os.mkdir(args.savepath)
    if not os.path.exists(savepath_folder):
        os.mkdir(savepath_folder)

    train_supervise_loss = train(model, dataset,supervise_epochs = args.epochs,
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



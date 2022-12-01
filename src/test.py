from evaluation_metric.measure_average_precision import * 
from evaluation_metric.f1_metric import *
from evaluation_metric.measure_coverage import * 
from evaluation_metric.measure_example_auc import *
from evaluation_metric.measure_macro_auc import * 
from evaluation_metric.measure_ranking_loss import * 
from evaluation_metric.measure_micro_auc import * 
from evaluation_metric.roc_auc import *

from tqdm import tqdm

from torch.autograd import Variable
import torch.nn as nn


def test_multilabel(args,Textfeaturemodel, Imgpredictmodel, Textpredictmodel, Imgmodel, Predictmodel, Attentionmodel, testdataset, batchsize = 32, cuda = False):
    if cuda:
        Textfeaturemodel.cuda()
        if Imgpredictmodel is not None:
            Imgpredictmodel.cuda()
            Textpredictmodel.cuda()
        Imgmodel.cuda()
        Predictmodel.cuda()
        Attentionmodel.cuda()
    Textfeaturemodel.eval()
    if Imgpredictmodel is not None:
        Imgpredictmodel.eval()
        Textpredictmodel.eval()
    Imgmodel.eval()
    Predictmodel.eval()
    Attentionmodel.eval()

    print('----------------- Test data:------------')
    
    total_predict = []
    img_predict = []
    text_predict = []
    truth = []
    sigmoid = torch.nn.Sigmoid()
    # data_loader = DataLoader(dataset = testdataset, batch_size = batchsize, shuffle = False)
    # for batch_index, (x, y) in enumerate(data_loader, 1):
    for batch_index, supbatch in tqdm(enumerate(testdataset), total=len(testdataset)):
        sup_img, sup_text, sup_label = supbatch

        # for single label only
        # sup_label = sup_label.unsqueeze(-1) if len(sup_label.shape) == 1 else sup_label # expand last dim for single label only
        # sup_label = torch.stack([1-sup_label, sup_label], axis=-1)

        img_xx = sup_img
        clip_ids = sup_text
        clip_ids = clip_ids.cuda() if cuda else clip_ids
        y = sup_label.numpy()
        img_xx = img_xx.float()
        img_xx = img_xx.cuda() if cuda else img_xx       
        with torch.no_grad():
            img_feature, img_encoded = Imgmodel(img_xx)
            text_feature,text_encoded = Textfeaturemodel(clip_emb = clip_ids)
            imghidden = torch.cat((img_feature, img_encoded),dim=1)
            texthidden = torch.cat((text_feature,text_encoded),dim=1)
            if args.use_deep_weak_attention:
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
                feature_hidden = img_attention * imghidden + text_attention * texthidden
            elif args.use_concat_modalities:
                feature_hidden = torch.cat((img_feature,text_feature,img_encoded,text_encoded), dim=1)
            imgpredict = sigmoid(Imgpredictmodel(imghidden))
            textpredict = sigmoid(Textpredictmodel(texthidden))
            predict = sigmoid(Predictmodel(feature_hidden))
        img_ = imgpredict.cpu().data.numpy()
        text_ = textpredict.cpu().data.numpy()
        img_predict.append(img_)
        text_predict.append(text_)
        predict = predict.cpu().data.numpy()
        total_predict.append(predict)

        truth.append(y)

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

    f1_weighted_multi_1  = weighted_f1_multilabel(total_predict, truth, num_labels=4, threshold = 0.5)
    f1_weighted_multi_2  = weighted_f1_multilabel(img_predict, truth, num_labels=4, threshold = 0.5)
    f1_weighted_multi_3  = weighted_f1_multilabel(text_predict, truth, num_labels=4, threshold = 0.5)

    # f1_skl1 = f1_score_sklearn(total_predict, truth)
    # f1_skl2 = f1_score_sklearn(img_predict, truth)
    # f1_skl3 = f1_score_sklearn(text_predict, truth)

    # f1_pm1 = f1_score_pytorch(total_predict, truth)
    # f1_pm2 = f1_score_pytorch(img_predict, truth)
    # f1_pm3 = f1_score_pytorch(text_predict, truth)

    auc_pm1 = auroc_score_pytorch(total_predict, truth)
    auc_pm2 = auroc_score_pytorch(img_predict, truth)
    auc_pm3 = auroc_score_pytorch(text_predict, truth)

    # average_precison1 = average_precision(total_predict, truth)
    # average_precison2 = average_precision(img_predict, truth)
    # average_precison3 = average_precision(text_predict, truth)
    
    # coverage1 = coverage(total_predict, truth)
    # coverage2 = coverage(img_predict, truth)
    # coverage3 = coverage(text_predict, truth)
    
    # example_auc1 = example_auc(total_predict, truth)
    # example_auc2 = example_auc(img_predict, truth)
    # example_auc3 = example_auc(text_predict, truth)

    # macro_auc1 = macro_auc(total_predict, truth)
    # macro_auc2 = macro_auc(img_predict, truth)
    # macro_auc3 = macro_auc(text_predict, truth)

    # micro_auc1 = micro_auc(total_predict, truth)
    # micro_auc2 = micro_auc(img_predict, truth)
    # micro_auc3 = micro_auc(text_predict, truth)

    # ranking_loss1 = ranking_loss(total_predict, truth)
    # ranking_loss2 = ranking_loss(img_predict, truth)
    # ranking_loss3 = ranking_loss(text_predict, truth)
    
    humour = np.histogram(total_predict[:,0])
    sarcasm = np.histogram(total_predict[:,1])
    offensive = np.histogram(total_predict[:,2])
    motivational = np.histogram(total_predict[:,3])

    humour_truth = np.histogram(truth[:,0])
    sarcasm_truth = np.histogram(truth[:,1])
    offensive_truth = np.histogram(truth[:,2])
    motivational_truth = np.histogram(truth[:,3])

    return (f1_macro_multi_1, f1_macro_multi_2, f1_macro_multi_3,
        f1_weighted_multi_1,f1_weighted_multi_2,f1_weighted_multi_3,
        auc_pm1,auc_pm2,auc_pm3,
        total_predict, truth, humour,sarcasm,offensive,motivational,humour_truth,
        sarcasm_truth,offensive_truth,motivational_truth)

def test_multilabel_finetune(args, model, testdataset, batchsize = 32, cuda = False):
    
    if cuda:
        model.cuda()
    model.eval()

    print('----------------- Test data:------------')
    
    total_predict = []
    img_predict = []
    text_predict = []
    truth = []
    sigmoid = torch.nn.Sigmoid()
    # data_loader = DataLoader(dataset = testdataset, batch_size = batchsize, shuffle = False)
    # for batch_index, (x, y) in enumerate(data_loader, 1):
    for ii, (supbatch) in tqdm(enumerate(testdataset), total = len(testdataset)):
        sup_img, sup_text, sup_label = supbatch

        # for single label only
        # sup_label = sup_label.unsqueeze(-1) if len(sup_label.shape) == 1 else sup_label # expand last dim for single label only
        # sup_label = torch.stack([1-sup_label, sup_label], axis=-1)

        y = sup_label.numpy()
        # yA = sup_label[:,0]
        # yB = sup_label[:,1:]
        
        image_feature = sup_img.float()
        text_feature = sup_text.float()
        
        if cuda:
            image_feature = image_feature.cuda()
            text_feature = text_feature.cuda()
        
        with torch.no_grad():
            pred = model(image_feature, text_feature)
            predict = sigmoid(pred)
            # predictB = sigmoid(predB)
            
        predict = predict.cpu().data.numpy()
        # predictB = predictB.cpu().data.numpy()
        total_predict.append(predict)
        # total_predictB.append(predictB)
        # truthA.append(yA.detach().cpu().numpy())
        truth.append(y)

    # total_predictA = np.array(total_predictA)
    # truthA = np.array(truthA)

    temp = total_predict[0]
    for i in range(1, len(total_predict)):
        temp = np.vstack((temp, total_predict[i]))
    total_predict = temp

    temp = truth[0]
    for i in range(1, len(truth)):
        temp = np.vstack((temp, truth[i]))
    truth = temp

    # temp = total_predictA[0]
    # for i in range(1, len(total_predictA)):
    #     temp = np.vstack((temp, total_predictA[i]))
    # total_predictA = temp
    # total_predictA = total_predictA.reshape(-1)
    # truthA = truthA.reshape(-1)
    # temp = truthA[0]
    # for i in range(1, len(truthA)):
    #     temp = np.vstack((temp, truthA[i]))
    # truthA = temp

    # macro_f1 = F1Score(num_classes=2, average='macro')
    # total_predictA = total_predictA > 0.5
    # f1_macro_total_A = macro_f1(torch.tensor(total_predictA).long(), torch.tensor(truthA).long())
    # result = {}

    f1_macro_multi_total = macro_f1_multilabel(total_predict, truth, num_labels=4, threshold = 0.5, reduce = True)

    # result.update({
    #     'f1_macro_multi_total': f1_macro_multi_total,
    #     'f1_macro_multi_img': f1_macro_multi_img,
    #     'f1_macro_multi_text': f1_macro_multi_text
    # })

    f1_weighted_multi_total = weighted_f1_multilabel(total_predict, truth, num_labels=4, threshold = 0.5)

    # result.update({
    #     'f1_weighted_multi_total': f1_weighted_multi_total,
    #     'f1_weighted_multi_img': f1_weighted_multi_img,
    #     'f1_weighted_multi_text': f1_weighted_multi_text
    # })

    # f1_weighted_multi_1  = weighted_f1_multilabel(total_predict[:,1:], truth[:,1:], num_labels=4, threshold = 0.5)
    # f1_weighted_multi_2  = weighted_f1_multilabel(img_predict[:,1:], truth[:,1:], num_labels=4, threshold = 0.5)
    # f1_weighted_multi_3  = weighted_f1_multilabel(text_predict[:,1:], truth[:,1:], num_labels=4, threshold = 0.5)

    # f1_skl1 = f1_score_sklearn(total_predict, truth)
    # f1_skl2 = f1_score_sklearn(img_predict, truth)
    # f1_skl3 = f1_score_sklearn(text_predict, truth)

    # f1_pm1 = f1_score_pytorch(total_predict, truth)
    # f1_pm2 = f1_score_pytorch(img_predict, truth)
    # f1_pm3 = f1_score_pytorch(text_predict, truth)

    # auc_pm_total = auroc_score_pytorch(total_predictB, truthB,num_labels=4)

    # result.update({
    #     'auc_pm_total': auc_pm_total,
    #     'auc_pm_img': auc_pm_img,
    #     'auc_pm_text': auc_pm_text
    # })

    # average_precison1 = average_precision(total_predict, truth)
    # average_precison2 = average_precision(img_predict, truth)
    # average_precison3 = average_precision(text_predict, truth)
    
    # coverage1 = coverage(total_predict, truth)
    # coverage2 = coverage(img_predict, truth)
    # coverage3 = coverage(text_predict, truth)
    
    # example_auc1 = example_auc(total_predict, truth)
    # example_auc2 = example_auc(img_predict, truth)
    # example_auc3 = example_auc(text_predict, truth)

    # macro_auc1 = macro_auc(total_predict, truth)
    # macro_auc2 = macro_auc(img_predict, truth)
    # macro_auc3 = macro_auc(text_predict, truth)

    # micro_auc1 = micro_auc(total_predict, truth)
    # micro_auc2 = micro_auc(img_predict, truth)
    # micro_auc3 = micro_auc(text_predict, truth)

    # ranking_loss1 = ranking_loss(total_predict, truth)
    # ranking_loss2 = ranking_loss(img_predict, truth)
    # ranking_loss3 = ranking_loss(text_predict, truth)
    
    # humour = np.histogram(total_predict[:,0])
    # sarcasm = np.histogram(total_predict[:,1])
    # offensive = np.histogram(total_predict[:,2])
    # motivational = np.histogram(total_predict[:,3])

    # humour_truth = np.histogram(truth[:,0])
    # sarcasm_truth = np.histogram(truth[:,1])
    # offensive_truth = np.histogram(truth[:,2])
    # motivational_truth = np.histogram(truth[:,3])

    # result.update({
    #     'hist_pred_1st' : humour,
    #     'hist_pred_2nd' : sarcasm,
    #     'hist_pred_3rd' : offensive,
    #     'hist_pred_4th' : motivational
    # })

    # result.update({
    #     'hist_truth_1st' : humour_truth,
    #     'hist_truth_2nd' : sarcasm_truth,
    #     'hist_truth_3rd' : offensive_truth,
    #     'hist_truth_4th' : motivational_truth
    # })

    # result.update({
    #     'truth': truth,
    #     'pred': total_predict
    # })
    auc_pm_total = None
    return (f1_macro_multi_total, f1_weighted_multi_total,auc_pm_total)

def test_singlelabel(args,Textfeaturemodel, Imgpredictmodel, Textpredictmodel, Imgmodel, Predictmodel, Attentionmodel, testdataset, batchsize = 32, cuda = False):
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
    total_predict = []
    img_predict = []
    text_predict = []
    truth = []
    # data_loader = Data.DataLoader(dataset = testdataset, batch_size = batchsize, shuffle = False)
    # for batch_index, (x, y) in enumerate(data_loader, 1):
    for batch_index, supbatch in tqdm(enumerate(testdataset), total=len(testdataset)):
        (sup_img, sup_text), sup_label = supbatch

        # sup_label = sup_label.unsqueeze(-1) if len(sup_label.shape) == 1 else sup_label # expand last dim for single label only
        sup_label = torch.stack([1-sup_label, sup_label], axis=-1)

        img_xx = sup_img
        text_xx = sup_text['sentence_vectors']
        if args.use_bert_embedding:
            bert_xx = sup_text['sbert_embedding']
        y = sup_label.numpy()

        img_xx = img_xx.float()
        text_xx = text_xx.float()
        if args.use_bert_embedding:
            bert_xx = bert_xx.float()
        img_xx = Variable(img_xx).cuda() if cuda else Variable(img_xx)
        text_xx = Variable(text_xx).cuda() if cuda else Variable(text_xx)
        if args.use_bert_embedding:
            bert_xx = Variable(bert_xx).cuda() if cuda else Variable(bert_xx)
        imghidden = Imgmodel(img_xx)
        if args.use_bert_embedding:
            texthidden = Textfeaturemodel(text_xx,bert_xx)
        else:
            texthidden = Textfeaturemodel(text_xx)

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
        #imgpredict = Predictmodel(imghidden)
        #textpredict = Predictmodel(texthidden)
        img_ = imgpredict.cpu().data.numpy()
        text_ = textpredict.cpu().data.numpy()
        predict = predict.cpu().data.numpy()
        total_predict.append(predict)
        img_predict.append(img_)
        text_predict.append(text_)
        truth.append(y)


        #if batch_index >= 2:
        #    break
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
    # print("---type: " , type(total_predict))
    # print("---type: " , type(truth))
    # print("---total_predict: " , total_predict)
    # print("---truth: " , truth)

    # f1_macro_multi_1 = macro_f1_multilabel(total_predict, truth, num_labels=4, threshold = 0.5, reduce = True)
    # f1_macro_multi_2 = macro_f1_multilabel(img_predict, truth, num_labels=4, threshold = 0.5, reduce = True)
    # f1_macro_multi_3 = macro_f1_multilabel(text_predict, truth,  num_labels=4, threshold = 0.5, reduce = True)

    # f1_skl1 = f1_score_sklearn(total_predict, truth)
    # f1_skl2 = f1_score_sklearn(img_predict, truth)
    # f1_skl3 = f1_score_sklearn(text_predict, truth)

    # f1_pm1 = f1_score_pytorch(total_predict, truth)
    # f1_pm2 = f1_score_pytorch(img_predict, truth)
    # f1_pm3 = f1_score_pytorch(text_predict, truth)

    # auc_pm1 = auroc_score_pytorch(total_predict, truth)
    # auc_pm2 = auroc_score_pytorch(img_predict, truth)
    # auc_pm3 = auroc_score_pytorch(text_predict, truth)
    

    macro_f1_all = macro_f1(total_predict[:,1:], truth[:,1:], threshold = 0.5)
    macro_f1_image = macro_f1(img_predict[:,1:], truth[:,1:], threshold = 0.5)
    macro_f1_text = macro_f1(text_predict[:,1:], truth[:,1:], threshold = 0.5)
    # average_precison1 = average_precision(total_predict, truth)
    # average_precison2 = average_precision(img_predict, truth)
    # average_precison3 = average_precision(text_predict, truth)
    
    # coverage1 = coverage(total_predict, truth)
    # coverage2 = coverage(img_predict, truth)
    # coverage3 = coverage(text_predict, truth)
    
    # example_auc1 = example_auc(total_predict, truth)
    # example_auc2 = example_auc(img_predict, truth)
    # example_auc3 = example_auc(text_predict, truth)

    macro_roc_auc1 = roc_auc_binary(total_predict, truth)
    macro_roc_auc2 = roc_auc_binary(img_predict, truth)
    macro_roc_auc3 = roc_auc_binary(text_predict, truth)

    # micro_auc1 = micro_auc(total_predict, truth)
    # micro_auc2 = micro_auc(img_predict, truth)
    # micro_auc3 = micro_auc(text_predict, truth)

    # ranking_loss1 = ranking_loss(total_predict, truth)
    # ranking_loss2 = ranking_loss(img_predict, truth)
    # ranking_loss3 = ranking_loss(text_predict, truth)
    
    misogynous = np.histogram(total_predict[:,0])
    # sarcasm = np.histogram(total_predict[:,1])
    # offensive = np.histogram(total_predict[:,2])
    # motivational = np.histogram(total_predict[:,3])

    misogynous_truth = np.histogram(truth[:,0])
    # sarcasm_truth = np.histogram(truth[:,1])
    # offensive_truth = np.histogram(truth[:,2])
    # motivational_truth = np.histogram(truth[:,3])

    return  macro_f1_all, macro_f1_image, macro_f1_text, macro_roc_auc1, macro_roc_auc2, macro_roc_auc3, total_predict, truth, misogynous, misogynous_truth

def test_auto_encoder(model, testdataset):
    model.cuda()
    model.eval()

    loss_func = nn.MSELoss()
    epoch_loss = 0
    for ii, (image_feature, text_feature, target) in tqdm(enumerate(testdataset), total = len(testdataset)):
        image_feature = image_feature.cuda().float()
        text_feature = text_feature.cuda().float()
        y = target.numpy()
        target = target.cuda().float()

        if model.encode_image:
            text_feature_pred = model(image_feature)                
            loss = loss_func(text_feature_pred, text_feature)
        elif model.encode_text:
            image_feature_pred = model(text_feature)                
            loss = loss_func(image_feature_pred, image_feature)
        
        epoch_loss += loss.item()

    return epoch_loss/len(testdataset)
    

def texttest(args,Textfeaturemodel, Textpredictmodel, testdataset, batchsize = 32, cuda = False):
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
        if args.use_bert_embedding:
            bert_xx = x[2]
        label = y.numpy()
        text_xx = text_xx.float()
        if args.use_bert_embedding:
            bert_xx = bert_xx.float()
        text_xx = Variable(text_xx).cuda() if cuda else Variable(text_xx)
        if args.use_bert_embedding:
            bert_xx = Variable(bert_xx).cuda() if cuda else Variable(bert_xx)
            textxx = Textfeaturemodel(text_xx,bert_xx)
        else:
            textxx = Textfeaturemodel(text_xx)
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

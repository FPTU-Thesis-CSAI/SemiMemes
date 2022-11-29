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
from src.model.AutoEncoder import AutoEncoder
from pretrain import train_auto_encoder 
from test import test_auto_encoder
from data.create_freq_file import dump_freq_data

def train(args,model, dataset,
        supervise_epochs = 200, text_supervise_epochs = 50, img_supervise_epochs = 50, 
        lr_supervise = 0.01, text_lr_supervise = 0.0001, img_lr_supervise = 0.0001,
        weight_decay = 0, batchsize = 32,lambda1=0.01,lambda2=1, textbatchsize = 32,
        imgbatchsize = 32, cuda = False, savepath = '',eman=None): 
    
    model.train()
    print("train")
    loss = 0
    cita = 1.003

    if args.use_sgd:
        print("==============use sgd optimizer===============")
        optimizer = torch.optim.SGD(model.parameters(), lr=lr_supervise, weight_decay=weight_decay)
    elif args.use_adam:
        print("==============use adam optimizer===============")
        optimizer = optim.Adam(model.parameters(), lr = lr_supervise, weight_decay = weight_decay)
    
    print("==============use step scheduler===============")
    scheduler = StepLR(optimizer, step_size = 100, gamma = 0.9)  

    if args.use_bce_loss:
        print("==============use bce loss===============")
        criterion = torch.nn.BCELoss()
    elif args.use_asymmetric_loss:
        print("==============use asymmetric loss===============")
        criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
    elif args.use_resample_loss:
        print("==============use resample loss===============")
        criterion = ResampleLoss(args)

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

        num_steps = min(len(dataset['train_sup']), len(dataset['train_unsup']))
        data_loaders = zip(dataset['train_sup'], dataset['train_unsup'])

        for step, batch  in tqdm(enumerate(data_loaders, start=1)
                        ,desc=f'epoch {epoch}',total=num_steps):
            
            (supbatch, unsupbatch) = batch
            sup_img, sup_text, sup_label = supbatch
            (unsup_img, unsup_text) = unsupbatch

            scheduler.step()
            y = sup_label
            '''
            Attention architecture and use bceloss.
            '''
            supervise_img_xx = sup_img
            label = sup_label
            # print("===============use clip token================")
            supervise_clip_ids = sup_text
            supervise_clip_ids = supervise_clip_ids.cuda() if cuda else supervise_clip_ids 

            supervise_img_xx = supervise_img_xx.float()
            label = label.float()
            supervise_img_xx = supervise_img_xx.cuda() if cuda else supervise_img_xx                 
            label = Variable(label).cuda() if cuda else Variable(label)  
            supervise_img_feature,supervise_img_encoded = model.Imgmodel(supervise_img_xx)
            supervise_text_feature,supervise_text_encoded = model.Textfeaturemodel(clip_emb = supervise_clip_ids)
            
            supervise_imghidden = torch.cat([supervise_img_feature,supervise_img_encoded],dim=1)
            supervise_texthidden = torch.cat([supervise_text_feature,supervise_text_encoded],dim=1)
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
                supervise_feature_hidden = img_attention * supervise_imghidden + text_attention * supervise_texthidden
            elif args.use_concat_modalities:
                # print("===============use concat================")
                supervise_feature_hidden = torch.cat((supervise_img_feature, supervise_text_feature,supervise_img_encoded,supervise_text_encoded), dim=1)

            supervise_predict = model.Predictmodel(supervise_feature_hidden) 
            if args.use_bce_loss:
                supervise_predict = sigmoid(supervise_predict)
                totalloss = criterion(supervise_predict, label)
                supervise_textpredict = sigmoid(supervise_textpredict)
                supervise_imgpredict = sigmoid(supervise_imgpredict)                
                imgloss = criterion(supervise_imgpredict, label)
                textloss = criterion(supervise_textpredict, label)
            elif args.use_asymmetric_loss or args.use_resample_loss:
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
                supervise_loss = modality_weights[1]*imgloss + modality_weights[2]*textloss + modality_weights[0]*totalloss + 0.01*div
                if args.supervise_entropy_minimization:
                    supervise_loss += entropy_img+entropy_text+entropy_total
            elif args.use_org_weights:
                supervise_loss = 0.16314931594158397*imgloss + 0.2173853996481428*textloss + 0.6194652844102732*totalloss + 0.01*div
                if args.supervise_entropy_minimization:
                    supervise_loss += entropy_img+entropy_text+entropy_total
            else:
                supervise_loss = imgloss + textloss + 2.0*totalloss + 0.01* div

            epoch_total_loss_train += totalloss.item()
            epoch_img_loss_train += imgloss.item()
            epoch_text_loss_train += textloss.item() 
            
            #=======================================#

            # ================== UNSUPERVISE =================== # 
            unsupervise_img_xx = unsup_img                        
            unsupervise_clip_ids = unsup_text
            unsupervise_clip_ids = unsupervise_clip_ids.cuda() if cuda else unsupervise_clip_ids
            unsupervise_img_xx = unsupervise_img_xx.float()
            unsupervise_img_xx = unsupervise_img_xx.cuda() if cuda else unsupervise_img_xx     
            unsupervise_img_feature,unsupervise_img_encoded = model.Imgmodel(unsupervise_img_xx)
            unsupervise_text_feature,unsupervise_text_encoded = model.Textfeaturemodel(clip_emb = unsupervise_clip_ids)
            unsupervise_imghidden = torch.cat((unsupervise_img_feature,unsupervise_img_encoded),dim=1)
            unsupervise_texthidden = torch.cat((unsupervise_text_feature,unsupervise_text_encoded),dim=1)

            unsupervise_imgpredict = model.Imgpredictmodel(unsupervise_imghidden)
            unsupervise_textpredict = model.Textpredictmodel(unsupervise_texthidden)
            
            unsupervise_imgpredict_unsharp = sigmoid(unsupervise_imgpredict)
            unsupervise_textpredict_unsharp = sigmoid(unsupervise_textpredict)
            unsupervise_imgpredict = sigmoid(args.T*unsupervise_imgpredict)
            unsupervise_textpredict = sigmoid(args.T*unsupervise_textpredict)
            if args.unsupervise_entropy_minimization:
                    unsup_entropy_img = EntropyLoss(unsupervise_imgpredict_unsharp)
                    unsup_entropy_text = EntropyLoss(unsupervise_textpredict_unsharp)
            
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

            total_loss = supervise_loss +  unsupervise_loss

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

        #===================== Multilabel =================#

        (f1_macro_multi_total, f1_macro_multi_img, f1_macro_multi_text,
        f1_weighted_multi_total,f1_weighted_multi_img,f1_weighted_multi_text,
        auc_pm1,auc_pm2,auc_pm3,
        total_predict, truth,humour,sarcasm,offensive,motivational,humour_truth,
        sarcasm_truth,offensive_truth,motivational_truth) = test_multilabel(args,model.Textfeaturemodel,
        model.Imgpredictmodel, model.Textpredictmodel, model.Imgmodel,
        model.Predictmodel, model.Attentionmodel, dataset['val'], batchsize = batchsize, cuda = cuda)

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

        wandb.log({"learning rate/lr":scheduler.get_last_lr()[0]})

        wandb.log({"f1_macro_multi_total Val":f1_macro_multi_total})
        wandb.log({"f1_macro_multi_img Val":f1_macro_multi_img})
        wandb.log({"f1_macro_multi_text Val":f1_macro_multi_text})

        wandb.log({"f1_weighted_multi_total Val":f1_weighted_multi_total})
        wandb.log({"f1_weighted_multi_img Val":f1_weighted_multi_img})
        wandb.log({"f1_weighted_multi_text Val":f1_weighted_multi_text})
        
        print(f"[F1 Macro multilabel] Total Val: {f1_macro_multi_total} Image {f1_macro_multi_img} Text {f1_macro_multi_text}")
        print(f"[F1 weight multilabel] Total Val: {f1_weighted_multi_total} Image {f1_weighted_multi_img} Text {f1_weighted_multi_text}")


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

        (f1_macro_multi_total, f1_macro_multi_img, f1_macro_multi_text,
        f1_weighted_multi_total,f1_weighted_multi_img,f1_weighted_multi_text,
        auc_pm1,auc_pm2,auc_pm3,
        total_predict, truth,humour,sarcasm,offensive,motivational,humour_truth,
        sarcasm_truth,offensive_truth,motivational_truth) = test_multilabel(args,model.Textfeaturemodel,
        model.Imgpredictmodel, model.Textpredictmodel, model.Imgmodel,
        model.Predictmodel, model.Attentionmodel, dataset['test'], batchsize = batchsize, cuda = cuda)

        wandb.log({"f1_macro_multi_total_test":f1_macro_multi_total})
        wandb.log({"f1_macro_multi_img_test":f1_macro_multi_img})
        wandb.log({"f1_macro_multi_text_test":f1_macro_multi_text})

        wandb.log({"f1_weighted_multi_total_test":f1_weighted_multi_total})
        wandb.log({"f1_weighted_multi_img_test":f1_weighted_multi_img})
        wandb.log({"f1_weighted_multi_text_test":f1_weighted_multi_text})
        
    
        print(f"[F1 Macro multilabel] Total Test: {f1_macro_multi_total} Image {f1_macro_multi_img} Text {f1_macro_multi_text}")
        print(f"[F1 weight multilabel] Total Test: {f1_weighted_multi_total} Image {f1_weighted_multi_img} Text {f1_weighted_multi_text}")

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
        clip_model, _ = clip.load(clip_nms[args.clip_model],jit=False)
        clip_model = clip_model.float()
        input_resolution = clip_model.visual.input_resolution
        cdim = clip_dim[args.clip_model]
        clip_model.eval()
        del clip_model
        gc.collect()

    label_cols = ['shaming', 'stereotype', 'objectification', 'violence']

    train_supervise_path = '/home/fptu/viet/SSLMemes/data/MAMI_processed/train_labeled_ratio-0.05.csv'
    train_supervise_image_feature_path = '/home/fptu/viet/SSLMemes/data/MAMI_processed/clip_features/train_labeled_ratio-0.05/image_feature.txt'
    train_supervise_text_feature_path = '/home/fptu/viet/SSLMemes/data/MAMI_processed/clip_features/train_labeled_ratio-0.05/text_feature.txt'

    unsupervise_image_feature_path = '/home/fptu/viet/SSLMemes/data/MAMI_processed/clip_features/train_unlabeled_ratio-0.05/image_feature.txt'
    unsupervise_text_feature_path = '/home/fptu/viet/SSLMemes/data/MAMI_processed/clip_features/train_unlabeled_ratio-0.05/text_feature.txt'

    if args.use_resample_loss:
        args.freq_file = dump_freq_data(train_supervise_path, label_cols)

    unsupervised_pretrain_loader = create_dataloader_pre_extracted(args,
                                                    image_features_path=unsupervise_image_feature_path,
                                                    text_features_path=unsupervise_text_feature_path,
                                                    shuffle=True, batch_size=args.batchsize, 
                                                    # normalize=True, feature_stats=feature_stats
                                                    )
    finetune_supervised_loader = create_dataloader_pre_extracted(args,
                                                    image_features_path=train_supervise_image_feature_path,
                                                    text_features_path=train_supervise_text_feature_path,
                                                    is_labeled=True, label_path=train_supervise_path, label_cols=label_cols,
                                                    shuffle=True, batch_size=args.batchsize,
                                                    # normalize=True, feature_stats=feature_stats
                                                    )  

    val_loader = create_dataloader_pre_extracted(args,
                                image_features_path='/home/fptu/viet/SSLMemes/data/MAMI_processed/clip_features/val/image_feature.txt',
                                text_features_path='/home/fptu/viet/SSLMemes/data/MAMI_processed/clip_features/val/text_feature.txt',
                                is_labeled=True, label_path='/home/fptu/viet/SSLMemes/data/MAMI_processed/val.csv', label_cols=label_cols,
                                shuffle=False, batch_size=args.batchsize,
                                # normalize=Tru
                                # e, feature_stats=feature_stats
                                )
    
    test_loader = create_dataloader_pre_extracted(args,
                                image_features_path='/home/fptu/viet/SSLMemes/data/MAMI_processed/clip_features/test/image_feature.txt',
                                text_features_path='/home/fptu/viet/SSLMemes/data/MAMI_processed/clip_features/test/text_feature.txt',
                                is_labeled=True, label_path='/home/fptu/viet/SSLMemes/data/MAMI_processed/test.csv', label_cols=label_cols,
                                shuffle=False, batch_size=args.batchsize,
                                # normalize=Tru
                                # e, feature_stats=feature_stats
                                )
    image_ae = AutoEncoder(encode_image=True)
    text_ae = AutoEncoder(encode_text=True)
    if cuda:
        image_ae.cuda()
        text_ae.cuda()
    if args.pretrain_auto_encoder:
        list_train_loss, list_val_loss = train_auto_encoder(image_ae, unsupervised_pretrain_loader, val_loader, cuda=cuda, verbose=3)
        test_loss = test_auto_encoder(image_ae, test_loader)
        print()
        print(test_loss)
        # wandb.log({"test_loss image ae": test_loss})
        
        list_train_loss, list_val_loss = train_auto_encoder(text_ae, unsupervised_pretrain_loader, val_loader, cuda=cuda, verbose=3)
        test_loss = test_auto_encoder(text_ae, test_loader)
        print()
        print(test_loss)

    dataset = {'train_sup': finetune_supervised_loader,
                'train_unsup': unsupervised_pretrain_loader,
                'val': val_loader,
                'test': test_loader}


    model = CmmlModel(args,image_encoder=image_ae.encoder,text_encoder=text_ae.encoder,cdim = cdim)

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


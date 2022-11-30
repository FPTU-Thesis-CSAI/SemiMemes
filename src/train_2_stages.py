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
from test_func import test_multilabel
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
from model.AutoEncoder import AutoEncoder, ModelCombineAE, ModelConCat
from pretrain import train_auto_encoder
from test_func import test_auto_encoder, test_multilabel_finetune
from data.create_freq_file import dump_freq_data
from torch.utils.data import random_split


def finetune_supervised(args,model, dataset,
        supervise_epochs = 200, text_supervise_epochs = 50, img_supervise_epochs = 50, 
        lr_supervise = 0.01, text_lr_supervise = 0.0001, img_lr_supervise = 0.0001,
        weight_decay = 0, batchsize = 32,lambda1=0.01,lambda2=1, textbatchsize = 32,
        imgbatchsize = 32, cuda = False, savepath = '',eman=None): 
    
    print("============================ Fine tune ===================================")
    num_update_steps = 15*8000/batchsize # MAMI train contains 8000 samples
    print(f"Update model for: {num_update_steps} steps.")

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
    
    print(optimizer)
    
    if args.use_linear_scheduler:
        print("==============use linear lr scheduler===============")
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,start_factor=1./3,total_iters=80)
    elif args.use_step_lr:
        print("==============use step scheduler===============")
        # 0.1 data is 800 sample
        scheduler = StepLR(optimizer, step_size = 800/batchsize, gamma = 0.95)  
    elif args.use_multi_step_lr:
        print("==============use multi step scheduler===============")
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [5,10,15],gamma=0.5)
    print(scheduler)
    
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
    
    print(criterion)
        
    sigmoid = torch.nn.Sigmoid()

    for epoch in range(1, supervise_epochs + 1):
        model.train()
        epoch_supervise_loss_train = 0
        # epoch_div_train = 0 
        # epoch_unsupervise_loss_train = 0
        # epoch_v_supervise_loss_train = 0 
        # epoch_c_supervise_loss_train = 0
        # epoch_v_unsupervise_loss_train = 0
        # epoch_c_unsupervise_loss_train = 0
        # epoch_img_loss_train = 0
        # epoch_text_loss_train = 0
        # epoch_total_loss_train = 0
        # if args.use_sim_loss:
        #     epoch_i_supervise_loss_train = 0
        #     epoch_i_unsupervise_loss_train = 0            
        # num_supervise_sample = 0
        # num_unsupervise_sample = 0

        num_steps = len(dataset['train_sup'])

        for step, supbatch in tqdm(enumerate(dataset['train_sup'], start=1),desc=f'epoch {epoch}', total=num_steps):

            sup_img, sup_text, sup_label = supbatch

            # for single label
            # sup_label = sup_label.unsqueeze(-1) if len(sup_label.shape) == 1 else sup_label # expand last dim for single label only
            # sup_label = torch.stack([1-sup_label, sup_label], axis=-1)
            # for single label
            
            label = sup_label
            
            image_feature = sup_img.float()
            text_feature = sup_text.float()
            label = label.float()
            
            if cuda:
                image_feature = image_feature.cuda()
                text_feature = text_feature.cuda()
                label = label.cuda()

            supervise_predict = model(image_feature, text_feature)    
            
            if args.use_focal_loss or  args.use_bce_loss:
                supervise_predict = sigmoid(supervise_predict)               
                sigmoid_already = True
            elif args.use_zlpr_loss or args.use_asymmetric_loss or args.use_resample_loss:
                sigmoid_already = False
            
            totalloss = criterion(supervise_predict, label)

            supervise_loss = totalloss

            # # ================== UNSUPERVISE =================== #
            
            unsupervise_loss = 0
            total_loss = supervise_loss + unsupervise_loss
                
            epoch_supervise_loss_train += supervise_loss.item()
            # # epoch_unsupervise_loss_train += unsupervise_loss.item()
            # # ============= optimize ==============#

            
            optimizer.zero_grad()
            total_loss.backward()
            if args.use_clip_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            scheduler.step()       

        if args.use_multi_step_lr or args.use_linear_scheduler:
            scheduler.step()

        #===================== Multilabel =================#
        (f1_macro_multi_total, f1_weighted_multi_total, auc_pm1,
            total_predict, truth, 
            humour,sarcasm,offensive,motivational,
            humour_truth,sarcasm_truth,offensive_truth,motivational_truth) = test_multilabel_finetune(args,model, dataset['val'], batchsize = batchsize, cuda = cuda)
        
        total_step = num_steps
        epoch_supervise_loss_train = epoch_supervise_loss_train/total_step
        # epoch_div_train = epoch_div_train/total_step
        # epoch_unsupervise_loss_train = epoch_unsupervise_loss_train/total_step
        # epoch_img_loss_train = epoch_img_loss_train/total_step
        # epoch_text_loss_train = epoch_text_loss_train/total_step 
        # epoch_total_loss_train = epoch_total_loss_train/total_step


        # print("epoch_total_loss_train:",epoch_total_loss_train)
        # print("epoch_img_loss_train",epoch_img_loss_train,
        #     "\t epoch_text_loss_train:",epoch_text_loss_train,
        #     "\t epoch_total_loss_train:",epoch_total_loss_train)
        
        # print("epoch_supervise_loss_train:",epoch_supervise_loss_train,
        # '\t epoch_div_train:',epoch_div_train,'\t epoch_unsupervise_loss_train',epoch_unsupervise_loss_train)
        
        print("epoch_supervise_loss_train:", epoch_supervise_loss_train)
        wandb.log({"epoch_supervise_loss_train":epoch_supervise_loss_train})

        wandb.log({"learning rate/lr":scheduler.get_last_lr()[0]})
        wandb.log({"f1_macro_multi_total":f1_macro_multi_total})
        # wandb.log({"f1_macro_multi_img":f1_macro_multi_img})
        # wandb.log({"f1_macro_multi_text":f1_macro_multi_text})

        wandb.log({"f1_weighted_multi_total":f1_weighted_multi_total})
        # wandb.log({"f1_weighted_multi_img":f1_weighted_multi_img})
        # wandb.log({"f1_weighted_multi_text":f1_weighted_multi_text})
                
        # print(f"[F1 Macro multilabel] Total: {f1_macro_multi_total} Image {f1_macro_multi_img} Text {f1_macro_multi_text}")
        # print(f"[F1 weight multilabel] Total: {f1_weighted_multi_total} Image {f1_weighted_multi_img} Text {f1_weighted_multi_text}")

        print(f"[F1 Macro multilabel] Total: {f1_macro_multi_total}")
        print(f"[F1 weight multilabel] Total: {f1_weighted_multi_total}")

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

        # print('rocauc_pm:    ', auc_pm1,'\t', auc_pm2,'\t', auc_pm3)
        print('rocauc_pm:    ', auc_pm1)

        # if not args.use_one_head: 
        (f1_macro_multi_total, f1_weighted_multi_total, auc_pm1,
            total_predict, truth, 
            humour,sarcasm,offensive,motivational,
            humour_truth,sarcasm_truth,offensive_truth,motivational_truth) = test_multilabel_finetune(args,model, dataset['test'], batchsize = batchsize, cuda = cuda)
        # else:
        #     (f1_macro_multi_total,f1_weighted_multi_total,auc_pm1,
        #     total_predict, truth, humour,sarcasm,offensive,motivational,humour_truth,
        #     sarcasm_truth,offensive_truth,motivational_truth) = test_multilabel(args,model.Textfeaturemodel,
        #     None,None, model.Imgmodel,
        #     model.Predictmodel, model.Attentionmodel, dataset['test'], batchsize = batchsize, cuda = cuda)

        wandb.log({"f1_macro_multi_total_test":f1_macro_multi_total})
        # if not args.use_one_head:
        # wandb.log({"f1_macro_multi_img_test":f1_macro_multi_img})
        # wandb.log({"f1_macro_multi_text_test":f1_macro_multi_text})

        wandb.log({"f1_weighted_multi_total_test":f1_weighted_multi_total})
        # if not args.use_one_head:
        # wandb.log({"f1_weighted_multi_img_test":f1_weighted_multi_img})
        # wandb.log({"f1_weighted_multi_text_test":f1_weighted_multi_text})
        
        # if args.use_one_head:
        #     print(f"Test [F1 Macro multilabel] Total: {f1_macro_multi_total}")
        #     print(f"[F1 weight multilabel] Total: {f1_weighted_multi_total}")
        # else:
        #     print(f"Test [F1 Macro multilabel] Total: {f1_macro_multi_total} Image {f1_macro_multi_img} Text {f1_macro_multi_text}")
        #     print(f"[F1 weight multilabel] Total: {f1_weighted_multi_total} Image {f1_weighted_multi_img} Text {f1_weighted_multi_text}")

        num_update_steps -= num_steps
        if num_update_steps <= 0: break
    
    return 
100

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
    print(args)

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

    # input_resolution = None
    # clip_model = None
    # cdim = None
    # if args.use_clip:
    #     print("==============use clip model===============")
    #     if args.use_open_clip:
    #         clip_model, _, _ = open_clip.create_model_and_transforms(args.clip_model,pretrained=args.clip_pretrained)
    #         input_resolution = clip_model.visual.image_size[0]
    #     else:
    #         clip_model, _ = clip.load(clip_nms[args.clip_model],jit=False)
    #         clip_model = clip_model.float()
    #         input_resolution = clip_model.visual.input_resolution
        
    #     cdim = clip_dim[args.clip_model]
    #     clip_model.eval()

    # train_supervised_loader, train_unsupervised_loader, val_loader  \
    #     = create_semi_supervised_dataloaders(args,
    #     train_img_dir='data/MAMI_processed/images/train',
    #     train_labeled_csv='data/MAMI_processed/train_labeled_ratio-0.05.csv',
    #     train_unlabeled_csv='data/MAMI_processed/train_unlabeled_ratio-0.05.csv',
    #     val_img_dir = 'data/MAMI_processed/images/val',
    #     val_csv='data/MAMI_processed/val.csv',
    #     batch_size=args.batchsize, image_size=256,input_resolution=input_resolution)
    
    # test_loader = create_semi_supervised_test_dataloaders(args,
    #                                                     test_img_dir='data/MAMI_processed/images/test',
    #                                                     test_csv='data/MAMI_processed/test.csv',
    #                                                     batch_size=args.batchsize, 
    #                                                     image_size=256,
    #                                                     input_resolution=input_resolution)

    # dataset = {'train_sup': train_supervised_loader,
    #             'train_unsup': train_unsupervised_loader,
    #             'val': val_loader,
    #             'test': test_loader}


    # # model = CmmlModel(args)
    # model = CmmlModel(args,clip_model = clip_model,cdim = cdim)
    
    # if args.dual_stream:
    #     model = CmmlModel_v2(args)
    
    label_cols = ['shaming', 'stereotype', 'objectification', 'violence']
    
    train_supervise_path = 'data/MAMI_processed/train_labeled_ratio-0.3.csv'
    train_supervise_image_feature_path = 'data/MAMI_processed/clip_features/train_labeled_ratio-0.3/image_feature.txt'
    train_supervise_text_feature_path = 'data/MAMI_processed/clip_features/train_labeled_ratio-0.3/text_feature.txt'
    
    unsupervise_image_feature_path = 'data/MAMI_processed/clip_features/train_unlabeled_ratio-0.3/image_feature.txt'
    unsupervise_text_feature_path = 'data/MAMI_processed/clip_features/train_unlabeled_ratio-0.3/text_feature.txt'
    
    # feature_stats = compute_feature_stats(unsupervise_image_feature_path, unsupervise_text_feature_path)
    
    unsupervised_pretrain_loader = create_dataloader_pre_extracted(args,
                                                                image_features_path=unsupervise_image_feature_path,
                                                                text_features_path=unsupervise_text_feature_path,
                                                                shuffle=True, batch_size=args.batchsize, 
                                                                # normalize=True, feature_stats=feature_stats
                                                                # is_split=True, val_size=1000
                                                                )
    
    finetune_supervised_loader = create_dataloader_pre_extracted(args,
                                                                image_features_path=train_supervise_image_feature_path,
                                                                text_features_path=train_supervise_text_feature_path,
                                                                is_labeled=True, label_path=train_supervise_path, label_cols=label_cols,
                                                                shuffle=True, batch_size=args.batchsize,
                                                                # normalize=True, feature_stats=feature_stats
                                                                )      

    val_loader = create_dataloader_pre_extracted(args,
                                                image_features_path='data/MAMI_processed/clip_features/val/image_feature.txt',
                                                text_features_path='data/MAMI_processed/clip_features/val/text_feature.txt',
                                                is_labeled=True, label_path='data/MAMI_processed/val.csv', label_cols=label_cols,
                                                shuffle=False, batch_size=args.batchsize,
                                                # normalize=True, feature_stats=feature_stats
                                                )

    test_loader = create_dataloader_pre_extracted(args,
                                                image_features_path='data/MAMI_processed/clip_features/test/image_feature.txt',
                                                text_features_path='data/MAMI_processed/clip_features/test/text_feature.txt',
                                                is_labeled=True, label_path='data/MAMI_processed/test.csv', label_cols=label_cols,
                                                shuffle=False, batch_size=args.batchsize,
                                                # normalize=True, feature_stats=feature_stats
                                                )

    if args.use_resample_loss:
        args.freq_file = dump_freq_data(train_supervise_path, label_cols)
        
    # unsupervised_pretrain_loader, unsupervised_pretrain_loader_val = random_split(unsupervised_pretrain_loader, 
    #                                                                               [int(len(unsupervised_pretrain_loader)*0.9),
    #                                                                                len(unsupervised_pretrain_loader)-int(len(unsupervised_pretrain_loader)*0.9)],
    #                                                                               generator=torch.Generator().manual_seed(42))

    if args.pretrain_auto_encoder:
        image_ae = AutoEncoder(encode_image=True)
        text_ae = AutoEncoder(encode_text=True)
        if cuda:
            image_ae.cuda()
            text_ae.cuda()
        
        list_train_loss, list_val_loss = train_auto_encoder(image_ae, unsupervised_pretrain_loader, val_loader, cuda=cuda, verbose=3, pretrain_epochs=100)
        test_loss = test_auto_encoder(image_ae, test_loader)
        print()
        print(test_loss)
        # wandb.log({"test_loss image ae": test_loss})
        
        list_train_loss, list_val_loss = train_auto_encoder(text_ae, unsupervised_pretrain_loader, val_loader, cuda=cuda, verbose=3, pretrain_epochs=100)
        test_loss = test_auto_encoder(text_ae, test_loader)
        print()
        print(test_loss)
        # wandb.log({"test_loss text ae": test_loss})     
        
    else:
        image_ae = ...
        text_ae = ...
        
    
    model = ModelCombineAE(image_encoder=image_ae.encoder, text_encoder=text_ae.encoder)
    
    # model = ModelConCat()

    if cuda:
        model = model.cuda()

    savepath_folder = args.savepath+"/"+args.experiment+"/"
    if not os.path.exists(args.savepath):
        os.mkdir(args.savepath)
    if not os.path.exists(savepath_folder):
        os.mkdir(savepath_folder)
        
    dataset = {'train_sup': finetune_supervised_loader,
            'val': val_loader,
            'test': test_loader}

    finetune_supervised(args,model, dataset,supervise_epochs = args.epochs,
                                        # text_supervise_epochs = args.text_supervise_epochs, 
                                        # img_supervise_epochs = args.img_supervise_epochs,
                                        lr_supervise = args.lr_supervise, 
                                        # text_lr_supervise = args.text_lr_supervise, 
                                        # img_lr_supervise = args.img_lr_supervise,
                                        weight_decay = args.weight_decay, 
                                        batchsize = args.batchsize,
                                        # textbatchsize = args.textbatchsize,
                                        # imgbatchsize = args.imgbatchsize,
                                        cuda = cuda, savepath = savepath_folder,
                                        # lambda1=args.lambda1,lambda2=args.lambda2
                                        )


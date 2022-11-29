
from tqdm import tqdm 
from model.utils import adjust_learning_rate
from torch.autograd import Variable
import torch 
import torch.nn as nn 
import numpy as np
import gc  
from copy import deepcopy
from model.utils import LARS,adjust_learning_rate,exclude_bias_and_norm
from torch.optim.lr_scheduler import StepLR

def compute_generalization(loss_val_N,loss_val_init):
    return abs(loss_val_N-loss_val_init)

def compute_overfitting(loss_val,loss_train):
    return loss_val - loss_train

def compute_O(N_epoch_loss_train,N_epoch_loss_val,init_loss_train,init_loss_val):
    N_epoch_overfit = compute_overfitting(N_epoch_loss_val,N_epoch_loss_train)
    init_epoch_overfit = compute_overfitting(init_loss_val,init_loss_train)
    return N_epoch_overfit - init_epoch_overfit

def cal_loss_val(args,model,dataset,optimizer,scheduler,criterion,cuda,sigmoid,mode=''):
    model.eval()
    num_steps = len(dataset['val'])
    if mode == 'img':
        img_loss_val = 0 
        for step, supbatch in tqdm(enumerate(dataset['val'], start=1)):
            img, text, label = supbatch
            img_xx = img
            label = label
            img_xx = img_xx.float()
            label = label.float()
            img_xx = img_xx.cuda() if cuda else img_xx               
            label = Variable(label).cuda() if cuda else Variable(label)
            with torch.no_grad():  
                img_feature,img_encoded = model.Imgmodel(img_xx)
                imghidden = torch.cat([img_feature,img_encoded],dim=1)
                imgpredict = model.Imgpredictmodel(imghidden)
                if args.use_bce_loss:
                    imgpredict = sigmoid(imgpredict)                
                    imgloss = criterion(imgpredict, label)
                elif args.use_asymmetric_loss or args.use_resample_loss:
                    imgloss = criterion(imgpredict, label)
            img_loss_val += imgloss.item()
        return img_loss_val/num_steps

    if mode == 'text':
        text_loss_val = 0 
        for step, supbatch in tqdm(enumerate(dataset['val'], start=1)):
            img, text, label = supbatch
            label = label
            clip_ids = text
            clip_ids = clip_ids.cuda() if cuda else clip_ids
            label = label.float()
            label = Variable(label).cuda() if cuda else Variable(label)  
            with torch.no_grad():
                text_feature,text_encoded = model.Textfeaturemodel(clip_emb = clip_ids)
                texthidden = torch.cat([text_feature,text_encoded],dim=1)
                textpredict = model.Textpredictmodel(texthidden)
                if args.use_bce_loss:
                    textpredict = sigmoid(textpredict)   
                    textloss = criterion(textpredict, label)
                elif args.use_asymmetric_loss or args.use_resample_loss:
                    textloss = criterion(textpredict, label)
            text_loss_val += textloss.item() 
        return text_loss_val/num_steps
    
    if mode == 'total':
        total_loss_val = 0 
        for step, supbatch in tqdm(enumerate(dataset['val'], start=1)):
            img, text, label = supbatch
            y = label
            img_xx = img
            label = label
            clip_ids = text
            clip_ids = clip_ids.cuda() if cuda else clip_ids 

            img_xx = img_xx.float()
            label = label.float()
            img_xx = img_xx.cuda() if cuda else img_xx             
            label = Variable(label).cuda() if cuda else Variable(label)  
            with torch.no_grad():
                img_feature,img_encoded = model.Imgmodel(img_xx)
                text_feature,text_encoded = model.Textfeaturemodel(clip_emb = clip_ids)
                if args.use_deep_weak_attention:
                    pass
                    # imgk = model.Attentionmodel(imghidden)
                    # textk = model.Attentionmodel(texthidden)
                    # modality_attention = []
                    # modality_attention.append(imgk)
                    # modality_attention.append(textk)
                    # modality_attention = torch.cat(modality_attention, 1)
                    # modality_attention = nn.functional.softmax(modality_attention, dim = 1)
                    # img_attention = torch.zeros(1, len(y))
                    # img_attention[0] = modality_attention[:,0]
                    # img_attention = img_attention.t()
                    # text_attention = torch.zeros(1, len(y))
                    # text_attention[0] = modality_attention[:,1]
                    # text_attention = text_attention.t()
                    # if cuda:
                    #     img_attention = img_attention.cuda()
                    #     text_attention = text_attention.cuda()
                    # feature_hidden = img_attention * imghidden + text_attention * texthidden
                elif args.use_concat_modalities:
                    feature_hidden = torch.cat((img_feature,text_feature,img_encoded,text_encoded), dim=1)
                predict = model.Predictmodel(feature_hidden)       
                if  args.use_bce_loss:
                    predict = sigmoid(predict)
                    totalloss = criterion(predict, label)
                elif args.use_asymmetric_loss or args.use_resample_loss:
                    totalloss = criterion(predict, label)
            total_loss_val += totalloss.item()
        return total_loss_val/num_steps

def create_optimizer_and_scheduler(args,model):
    if args.use_sgd:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_supervise, weight_decay=args.weight_decay)
    elif args.use_adam:
        optimizer = torch.optim.Adam(model.parameters(), lr = args.lr_supervise, weight_decay = args.weight_decay)
    
    scheduler = StepLR(optimizer, step_size = 100, gamma = 0.9)  
    return optimizer, scheduler
    
def GB_estimate(args,orig_model,train_epochs,dataset,optimizer,scheduler,criterion,cuda,sigmoid):
    
    model = deepcopy(orig_model)
    optimizer, scheduler = create_optimizer_and_scheduler(args,model)
    num_steps = len(dataset['train_sup'])
    print("===========train image modality===========")
    for epoch in range(1,train_epochs + 1):
        epoch_img_loss_train = 0
        if epoch == 1:
            model.eval()
        else:
            model.train()
        for step, supbatch in enumerate(tqdm(dataset['train_sup'],total=len(dataset['train_sup']),desc=f'epoch {epoch}, modality image')):
            sup_img, sup_text, sup_label = supbatch
            if epoch != 1:
                scheduler.step()
            supervise_img_xx = sup_img
            label = sup_label
            supervise_img_xx = supervise_img_xx.float()
            label = label.float()
            supervise_img_xx = supervise_img_xx.cuda() if cuda else supervise_img_xx            
            label = Variable(label).cuda() if cuda else Variable(label)  
            supervise_img_feature,supervise_img_encoded = model.Imgmodel(supervise_img_xx)
            supervise_imghidden = torch.cat([supervise_img_feature,supervise_img_encoded],dim=1)
            supervise_imgpredict = model.Imgpredictmodel(supervise_imghidden)
            if args.use_bce_loss:
                supervise_imgpredict = sigmoid(supervise_imgpredict)
                imgloss = criterion(supervise_imgpredict, label)
            elif args.use_asymmetric_loss or args.use_resample_loss:
                imgloss = criterion(supervise_imgpredict, label)
            epoch_img_loss_train += imgloss.item()
            if epoch != 1:
                optimizer.zero_grad()
                imgloss.backward()
                if args.use_clip_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        if epoch==1:
            initial_img_loss_train = epoch_img_loss_train/num_steps
            initial_img_loss_val =  cal_loss_val(args,model,dataset,optimizer,scheduler,criterion,cuda,sigmoid,mode='img')
        elif epoch==train_epochs:
            N_epoch_img_loss_train = epoch_img_loss_train/num_steps
            N_epoch_img_loss_val = cal_loss_val(args,model,dataset,optimizer,scheduler,criterion,cuda,sigmoid,mode='img')
    print(f"initial_img_loss_train:{initial_img_loss_train}, initial_img_loss_val:{initial_img_loss_val}")
    print(f"N_epoch_img_loss_train: {N_epoch_img_loss_train}, N_epoch_img_loss_val:{N_epoch_img_loss_val}")

    del model
    gc.collect()
    print("==========train text modality============")
    model = deepcopy(orig_model)
    optimizer, scheduler = create_optimizer_and_scheduler(args,model)
    for epoch in range(1,train_epochs + 1):
        epoch_text_loss_train = 0
        if epoch == 1:
            model.eval()
        else:
            model.train()
        for step, supbatch in enumerate(tqdm(dataset['train_sup'],total=len(dataset['train_sup']),desc=f'epoch {epoch}, modality text')):
            sup_img, sup_text, sup_label = supbatch
            if epoch != 1:
                scheduler.step()
            label = sup_label
            supervise_clip_ids = sup_text
            supervise_clip_ids = supervise_clip_ids.cuda() if cuda else supervise_clip_ids 
            label = label.float()
            label = Variable(label).cuda() if cuda else Variable(label)  

            supervise_text_feature,supervise_text_encoded = model.Textfeaturemodel(clip_emb = supervise_clip_ids)
            supervise_texthidden = torch.cat([supervise_text_feature,supervise_text_encoded],dim=1)
            supervise_textpredict = model.Textpredictmodel(supervise_texthidden)
            if args.use_bce_loss:
                supervise_textpredict = sigmoid(supervise_textpredict)
                textloss = criterion(supervise_textpredict, label)
            elif args.use_asymmetric_loss or args.use_resample_loss:
                textloss = criterion(supervise_textpredict, label)
            epoch_text_loss_train += textloss.item()
            if epoch != 1:
                optimizer.zero_grad()
                textloss.backward()
                if args.use_clip_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        if epoch==1:
            initial_text_loss_train = epoch_text_loss_train/num_steps
            initial_text_loss_val =  cal_loss_val(args,model,dataset,optimizer,scheduler,criterion,cuda,sigmoid,mode='text')
        elif epoch==train_epochs:
            N_epoch_text_loss_train = epoch_text_loss_train/num_steps
            N_epoch_text_loss_val = cal_loss_val(args,model,dataset,optimizer,scheduler,criterion,cuda,sigmoid,mode='text')
    print(f"initial_img_loss_train:{initial_text_loss_train}, initial_img_loss_val:{initial_text_loss_val}")
    print(f"N_epoch_img_loss_train: {N_epoch_text_loss_train}, N_epoch_img_loss_val:{N_epoch_text_loss_val}")
    del model 
    gc.collect()

    print("===============train combine modality================")
    model = deepcopy(orig_model)
    optimizer, scheduler = create_optimizer_and_scheduler(args,model)
    for epoch in range(1,train_epochs + 1):
        epoch_total_loss_train = 0
        if epoch == 1:
            model.eval()
        else:
            model.train()
        for step, supbatch in enumerate(tqdm(dataset['train_sup'],total=len(dataset['train_sup']),desc=f'epoch {epoch}, modality combine')):
            sup_img, sup_text, sup_label = supbatch
            if epoch != 1:
                scheduler.step()
            y = sup_label
            '''
            Attention architecture and use bceloss.
            '''
            supervise_img_xx = sup_img
            label = sup_label

            supervise_clip_ids = sup_text
            supervise_clip_ids = Variable(supervise_clip_ids).cuda() if cuda else Variable(supervise_clip_ids)    

            supervise_img_xx = supervise_img_xx.float()
            label = label.float()
            supervise_img_xx = Variable(supervise_img_xx).cuda() if cuda else Variable(supervise_img_xx)                  
            label = Variable(label).cuda() if cuda else Variable(label)  
            supervise_img_feature,supervise_img_encoded = model.Imgmodel(supervise_img_xx)
            supervise_text_feature,supervise_text_encoded = model.Textfeaturemodel(clip_emb = supervise_clip_ids)

            if args.use_deep_weak_attention:
                pass
                # supervise_imgk = model.Attentionmodel(supervise_imghidden)
                # supervise_textk = model.Attentionmodel(supervise_texthidden)
                # modality_attention = []
                # modality_attention.append(supervise_imgk)
                # modality_attention.append(supervise_textk)
                # modality_attention = torch.cat(modality_attention, 1)
                # modality_attention = nn.functional.softmax(modality_attention, dim = 1)
                # img_attention = torch.zeros(1, len(y))
                # img_attention[0] = modality_attention[:,0]
                # img_attention = img_attention.t()
                # text_attention = torch.zeros(1, len(y))
                # text_attention[0] = modality_attention[:,1]
                # text_attention = text_attention.t()
                # if cuda:
                #     img_attention = img_attention.cuda()
                #     text_attention = text_attention.cuda()
                # supervise_feature_hidden = img_attention * supervise_imghidden + text_attention * supervise_texthidden
            elif args.use_concat_modalities:
                supervise_feature_hidden = torch.cat((supervise_img_feature, supervise_text_feature,supervise_img_encoded,supervise_text_encoded), dim=1)
            supervise_predict = model.Predictmodel(supervise_feature_hidden)       
            if args.use_bce_loss:
                supervise_predict = sigmoid(supervise_predict)
                totalloss = criterion(supervise_predict, label)
            elif args.use_asymmetric_loss or args.use_resample_loss:
                totalloss = criterion(supervise_predict, label)

            epoch_total_loss_train += totalloss.item()
            optimizer.zero_grad()
            if epoch != 1:
                totalloss.backward()
                if args.use_clip_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        if epoch==1:
            initial_total_loss_train = epoch_total_loss_train/num_steps 
            initial_total_loss_val =  cal_loss_val(args,model,dataset,optimizer,scheduler,criterion,cuda,sigmoid,mode='total')
        elif epoch==train_epochs:
            N_epoch_total_loss_train = epoch_total_loss_train/num_steps 
            N_epoch_total_loss_val =  cal_loss_val(args,model,dataset,optimizer,scheduler,criterion,cuda,sigmoid,mode='total')
    print(f"initial_img_loss_train:{initial_total_loss_train}, initial_img_loss_val:{initial_total_loss_val}")
    print(f"N_epoch_img_loss_train: {N_epoch_total_loss_train}, N_epoch_img_loss_val:{N_epoch_total_loss_val}")
    del model 
    gc.collect()

    total_gen = compute_generalization(N_epoch_total_loss_val,initial_total_loss_val)
    img_gen = compute_generalization(N_epoch_img_loss_val,initial_img_loss_val)
    text_gen = compute_generalization(N_epoch_text_loss_val,initial_text_loss_val)
    total_o = compute_O(N_epoch_total_loss_train,N_epoch_total_loss_val,initial_total_loss_train,initial_total_loss_val)
    img_o = compute_O(N_epoch_img_loss_train,N_epoch_img_loss_val,initial_img_loss_train,initial_img_loss_val)
    text_o = compute_O(N_epoch_text_loss_train,N_epoch_text_loss_val,initial_text_loss_train,initial_text_loss_val)
    if args.original_org:
        print("use original ORG")
        coeff = np.array([total_gen/total_o**2,img_gen/img_o**2,text_gen/text_o**2])
        return coeff/sum(coeff)
    elif args.modified_org:
        print("use modified ORG")
        coeff = np.array([1./(total_gen*total_o**2),1./(img_gen*img_o**2),1./(text_gen*text_o**2)])
        return coeff/sum(coeff)


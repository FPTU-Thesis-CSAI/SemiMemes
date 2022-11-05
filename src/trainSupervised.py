from arguments import get_args 
import wandb 
from model.unsupervised import UnsupervisedModel 
import torch 
from data.dataClass import get_supervision_dataset_hateful,get_supervision_dataset_harmeme,get_supervision_dataset_memotion
from model.classifier import MultiModalClassifier 
from unsupervisedUtils import save_config_file,save_checkpoint
import os  
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm 
from supervisedUtils import plot_confusion_matrix 

def train_on_one_epoch(args,model,classifier,train_loader,optimizer,trainloss,total_preds,total_true):
    for train_loader_idx, (img1, text, mask, labels) in enumerate(train_loader):

        if args.dryrun and train_loader_idx >= 4:
            print("Dry Run in train complete, exiting")
            break

        bs = img1.shape[0]
        img1 = img1.to(args.device)
        text = text.to(args.device)
        mask = mask.to(args.device)
        labels = labels.to(args.device)
        with autocast(enabled=args.fp16_precision):
            with torch.no_grad():
                txt_repr = model.text_encoder(input_ids=text, attention_mask=mask)
                img_feats = model.image_encoder(img1)
            out = classifier(img_feats, txt_repr)
            loss = criterion(out, labels)
            total_preds = torch.cat((total_preds, out.cpu().detach()), dim=0)
            total_true = torch.cat((total_true, labels.cpu()), dim=0)
            trainloss += loss.item()

        optimizer.zero_grad()

        scaler.scale(loss).backward()

        scaler.step(optimizer)
        scaler.update()
    return trainloss,total_preds,total_true
    
def train(args,model,classifier,train_loader,val_loader,optimizer):
    best_val_f1 = 0.0
    corr_train_f1 = 0.0
    best_val_auroc = 0.0
    corr_train_auroc = 0.0
    for epoch_counter in tqdm(range(args.epochs), disable=args.no_tqdm):
        trainloss = 0
        correct = 0
        total_preds = torch.tensor([])
        total_true = torch.tensor([])
        classifier.train()

        trainloss,total_preds,total_true = train_on_one_epoch(args,model,classifier,train_loader,
                            optimizer,trainloss,total_preds,total_true)

        pos_probs = total_preds[:, 1]
        if args.num_classes == 2:
            train_auroc = roc_auc_score(total_true, pos_probs)
        if args.task in ['a', 'A']:
            total_preds = total_preds.argmax(dim=1)
            train_cm = confusion_matrix(total_true, total_preds)
        else:
            total_preds = (torch.sigmoid(total_preds) > 0.5)
        train_ac = accuracy_score(total_true, total_preds)
        train_f1 = f1_score(total_true, total_preds, average='macro')

        trainloss /= len(train_loader) 

        if epoch_counter >= 3 and scheduler is not None:
            scheduler.step()

        classifier.eval()
        with torch.no_grad():
            valloss = 0
            total_preds = torch.tensor([])
            total_true = torch.tensor([])
            for val_loader_idx, (img1, text, mask, labels) in enumerate(val_loader):

                if args.dryrun:
                    if val_loader_idx == 4:
                        print("Dry Run in val complete, exiting")
                        break

                bs = img1.shape[0]
                img1 = img1.to(args.device)
                text = text.to(args.device)
                mask = mask.to(args.device)
                text = torch.squeeze(text, 1)
                mask = torch.squeeze(mask, 1)
                labels = labels.to(args.device)
                
                with autocast(enabled=args.fp16_precision):
                    txt_repr = model.text_encoder(text, mask)
                    img1_feats = model.image_encoder(img1)
                    out = classifier(img1_feats, txt_repr)
                    total_preds = torch.cat((total_preds, out.cpu().detach()), dim=0)
                    total_true = torch.cat((total_true, labels.cpu()), dim=0)
                    loss = criterion(out, labels)
                    valloss += loss.item()

            valloss /= len(val_loader) 
            pos_probs = total_preds[:, 1]

            if args.num_classes == 2:
                val_auroc = roc_auc_score(total_true, pos_probs)
            if args.task in ['a', 'A']:
                total_preds = total_preds.argmax(dim=1)
                val_cm = confusion_matrix(total_true, total_preds)
            else:
                total_preds = (torch.sigmoid(total_preds) > 0.5)

            val_ac = accuracy_score(total_true, total_preds)
            val_f1 = f1_score(total_true, total_preds, average='macro')

        if args.num_classes == 2:
            if val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                corr_train_auroc = train_auroc

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            corr_train_f1 = train_f1

        if args.dryrun:
            break
        
        wandb.log({"epoch":epoch_counter})
        wandb.log({'training/loss': trainloss})
        wandb.log({'validation/loss': valloss})
        wandb.log({'learning_rate': scheduler.get_last_lr()[0]})
        wandb.log({'training/f1_score': train_f1})
        wandb.log({'validation/f1_score': val_f1})
        if args.num_classes == 2:
            wandb.log({'training/auc_roc': train_auroc})
            wandb.log({'validation/auc_roc': val_auroc})

        wandb.log({'training/accuracy': train_ac})
        wandb.log({'validation/accuracy': val_ac})
        if args.task in ['a', 'A']:
            train_cm_fig = plot_confusion_matrix(train_cm, ['0', '1', '2'])
            wandb.log({'training/confusion_matrix':train_cm_fig})
            val_cm_fig = plot_confusion_matrix(val_cm, ['0', '1', '2'])
            wandb.log({'validation/confusion_matrix': val_cm_fig})

        msg = f"Epoch: {epoch_counter}\tTrain Loss: {trainloss}\tValidation Loss: {valloss}"
        msg += f"\n-----:---\tTrain Accuracy: {train_ac}\tValidation Accuracy: {val_ac}"
        msg += f"\n-----:---\tTrain F1: {train_f1}\tValidation F1: {val_f1}"
        if args.num_classes == 2:
            msg += f"\n-----:---\tTrain AUROC: {train_auroc}\tValidation AUROC: {val_auroc}"
        print(msg) 

        if epoch_counter % 10 == 9:   
            checkpoint_name = '{}_{:04d}.pth.tar'.format(classifier.name, epoch_counter)
            save_checkpoint({
                'epoch': args.epochs,
                'arch': args.arch,
                'state_dict': classifier.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, is_best=False, filename=os.path.join(log_dir, checkpoint_name))
            print(f"Checkpoint created at {checkpoint_name}")

    # save model checkpoints
    checkpoint_name = 'last_checkpoint-{}--c--{}.pth.tar'.format(model.name, classifier.name)
    save_checkpoint({
        'epoch': args.epochs,
        'arch': args.arch,
        'state_dict': classifier.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
    print(f"Model checkpoint and metadata has been saved at {log_dir}.")
    msg = "Training F1:{:.4f} and Testset F1:{:.4f}".format(corr_train_f1, best_val_f1)
    if args.num_classes == 2:
        msg += " Training AUROC: {:.4f} and Testset AUROC: {:.4f}".format(corr_train_auroc, best_val_auroc)
    print("Training Completed")

def evaluateOnly(args,model,classifier,val_loader,log_dir,criterion):
    model.eval()
    save_config_file(log_dir, args)

    print(f"Start Supervised Evaluation for {args.epochs} epochs.")
    print(f"Using args: {args}")
    best_val_f1 = 0.0
    corr_train_f1 = 0.0
    best_val_auroc = 0.0
    corr_train_auroc = 0.0
    classifier.eval()
    with torch.no_grad():
        valloss = 0
        total_preds = torch.tensor([])
        total_true = torch.tensor([])
        for val_loader_idx, (img1, text, mask, labels) in enumerate(val_loader):
            bs = img1.shape[0]
            img1 = img1.to(args.device)
            text = text.to(args.device)
            mask = mask.to(args.device)
            text = torch.squeeze(text, 1)
            mask = torch.squeeze(mask, 1)
            labels = labels.to(args.device)
                
            with autocast(enabled=args.fp16_precision):
                txt_repr = model.text_encoder(text, mask)
                img1_feats = model.image_encoder(img1)
                out = classifier(img1_feats, txt_repr)
                total_preds = torch.cat((total_preds, out.cpu().detach()), dim=0)
                total_true = torch.cat((total_true, labels.cpu()), dim=0)
                loss = criterion(out, labels)
                valloss += loss.item()

        valloss /= len(val_loader) 
        pos_probs = total_preds[:, 1]

        if args.num_classes == 2:
            val_auroc = roc_auc_score(total_true, pos_probs)
        if args.task in ['a', 'A']:
            total_preds = total_preds.argmax(dim=1)
            val_cm = confusion_matrix(total_true, total_preds)
        else:
            total_preds = (torch.sigmoid(total_preds) > 0.5)

        val_ac = accuracy_score(total_true, total_preds)
        val_f1 = f1_score(total_true, total_preds, average='macro')
        if args.num_classes == 2:
            if val_auroc > best_val_auroc:
                best_val_auroc = val_auroc

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1

    msg = "Testset F1:{:.4f}".format(best_val_f1)
    if args.num_classes == 2:
        msg += " and AUROC: {:.4f}".format(best_val_auroc)
    print("Evaluation Completed")
    print(msg)

args = get_args()

if args.dryrun:
    args.experiment = 'dryrun'

assert (not args.simclr) or (args.n_views > 1), "SimCLR requires at least 2 image views"

if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')
    args.gpu_index = -1

print(args)
wandb.init(project="meme_experiments", entity="vietnguyen")
ckpt_use = args.ckpt != ''
model = UnsupervisedModel(args.arch, args.txtmodel, args.out_dim, args.dropout, args.projector, not ckpt_use, not ckpt_use)
model.to(args.device)
print(f"Unsupervised Model Name: {model.name}")

if ckpt_use:
    model.load_state_dict(torch.load(args.ckpt, map_location=args.device)['state_dict'])
    print(f"Model Loaded from {args.ckpt}")

if args.dataset_name == 'hatefulmemes':
    train_dataset, val_dataset = get_supervision_dataset_hateful(args)
elif args.dataset_name == 'harmeme':
    train_dataset, val_dataset = get_supervision_dataset_harmeme(args)
elif args.dataset_name == 'memotion':
    train_dataset, val_dataset = get_supervision_dataset_memotion(args)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True,
    num_workers=args.workers)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers)

classifier = MultiModalClassifier(
        fuse_type=args.fuse_type,
        model=model,
        num_classes=args.num_classes,
        nl=args.nl,
        bn=args.bn
).to(args.device)
print(f"Classification Head Name: {classifier.name}")

if args.cl_ckpt != '':
    classifier.load_state_dict(torch.load(args.cl_ckpt, map_location=args.device)['state_dict'])
    print(f"Classifier Loaded from {args.cl_ckpt}")

optimizer = torch.optim.Adam(classifier.parameters(), args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 1e-3,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs
)
pytorch_total_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
print("Total Parameters: ", pytorch_total_params)
print("="*35, "CLASSIFIER", "="*35)
print(classifier)
if args.task in ['a', 'A']:
    criterion = torch.nn.CrossEntropyLoss().to(args.device)
else:
    criterion = torch.nn.BCEWithLogitsLoss().to(args.device)

save_folder = '/home/viet/SSLMemes/saved_model/'
if not os.path.exists(save_folder):
    os.mkdir(save_folder)

log_dir = os.path.join(save_folder,args.experiment)
if not os.path.exists(log_dir):
    os.mkdir(log_dir)


if args.evaluate_only:
    evaluateOnly(args,model,classifier,val_loader,log_dir,criterion)
else:
    scaler = GradScaler(enabled=args.fp16_precision)
    model.eval()
    save_config_file(log_dir, args)
    print(f"Start Supervised training for {args.epochs} epochs.")
    print(f"Using args: {args}")
    train(args,model,classifier,train_loader,val_loader,optimizer)


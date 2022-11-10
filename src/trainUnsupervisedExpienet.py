from arguments import get_args 
from model.unsupervised import UnsupervisedModel
import torch  
from data.dataClass import get_unsupervision_dataset
import os 
import logging 
from loss import MMContrastiveLoss 
from lightly.loss import ntx_ent_loss
from utils.unsupervisedUtils import save_embed,save_config_file,get_batch,EarlyStopping,save_checkpoint,compute_loss
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm 
import wandb 

def train_on_one_epoch(args,model,train_loader,optimizer,scaler,n_iter,epoch_counter,meme_mmloss=None,meme_floss=None,mmcontr_loss=None):
    trainiterator = iter(train_loader)
    for loader_idx in range(len(train_loader)):
        batch = get_batch(args,trainiterator)

        if args.dryrun:
            if loader_idx == 4:
                print("Dry Run in Unsupervised train complete, exiting")
                break

        with autocast(enabled=args.fp16_precision):
            out = model(batch)
            if args.memeloss:
                loss = compute_loss(args,out,epoch_counter,meme_mmloss=meme_mmloss,meme_floss=meme_floss)
            elif args.mmcontr:
                loss = compute_loss(args,out,epoch_counter,mmcontr_loss=mmcontr_loss)

        optimizer.zero_grad()

        scaler.scale(loss).backward()

        scaler.step(optimizer)
        scaler.update()
        n_iter += 1
    return loss

def train(args,model,earlyStopper,train_loader,optimizer,scaler,n_iter,meme_mmloss=None,meme_floss=None,mmcontr_loss=None):
    for epoch_counter in tqdm(range(args.epochs), disable=args.no_tqdm):
        loss = train_on_one_epoch(args,model,train_loader,optimizer,scaler,n_iter,epoch_counter,meme_mmloss=meme_mmloss,meme_floss=meme_floss,mmcontr_loss=mmcontr_loss)
        # earlyStopper(loss.item())

        print("Epoch: {}\tLoss: {}".format(epoch_counter, loss.item()))
        wandb.log({"loss": loss.item()})
        wandb.log({"epoch": epoch_counter})
        if args.dryrun:
            break

        if epoch_counter >= 10:
            scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        wandb.log({"learning rate": lr})
        if epoch_counter % 5 == 0:
            checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(epoch_counter)
            save_checkpoint({
                'epoch': args.epochs,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, is_best=False, filename=os.path.join(log_dir, checkpoint_name))
            print(f"Checkpoint created at {checkpoint_name}")

        if earlyStopper.early_stop:
            print("Early Stopping, Loss didn't decrease for several epochs")
            break


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
wandb.init(project="meme_experiments", entity="meme-analysts",mode="disabled")
wandb.run.name = args.experiment
ckpt_use = args.ckpt != ''
model = UnsupervisedModel(args.arch, args.txtmodel, args.out_dim, args.dropout, args.projector, not ckpt_use, not ckpt_use)
model.to(args.device)
print(f"Unsupervised Model Name: {model.name}")

if ckpt_use:
    model.load_state_dict(torch.load(args.ckpt, map_location=args.device)['state_dict'])
    print(f"Model Loaded from {args.ckpt}")

train_dataset = get_unsupervision_dataset(args)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True,
    num_workers=args.workers, pin_memory=True)
optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                        last_epoch=-1)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total Parameters: ", pytorch_total_params)

save_folder = '/home/viet/SSLMemes/saved_model/'
if not os.path.exists(save_folder):
    os.mkdir(save_folder)

log_dir = os.path.join(save_folder,args.experiment)
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
mmcontr_loss,meme_mmloss,meme_floss = None, None, None 
with torch.cuda.device(args.gpu_index):
    if args.mmcontr:
        mmcontr_loss= MMContrastiveLoss(
                margin=args.margin,
                measure=args.measure,
                max_violation=args.max_violation
            ).to(args.device)
    if args.memeloss:
        meme_mmloss = MMContrastiveLoss(
                margin=args.margin,
                measure=args.measure,
                max_violation=args.max_violation
        ).to(args.device)
        meme_floss = ntx_ent_loss.NTXentLoss(args.temperature,args.moco_size).to(args.device)
    if args.vis_embed:
        log_dir = '/home/viet/SSLMemes/saved_model/'
        save_embed(args,model,train_loader,log_dir)
    else:
        scaler = GradScaler(enabled=args.fp16_precision)
        save_config_file(log_dir,args)
        n_iter = 0
        print(f"Start SimCLR training for {args.epochs} epochs.")
        print(f"Using args: {args}")

        earlyStopper = EarlyStopping(patience=10)
        train(args,model,earlyStopper,train_loader,optimizer,scaler,n_iter,meme_mmloss=meme_mmloss,meme_floss=meme_floss,mmcontr_loss=mmcontr_loss)
        print("Training has finished.")

        checkpoint_name = 'last_checkpoint-{}.pth.tar'.format(model.name)
        filename = os.path.join(log_dir, checkpoint_name)
        save_checkpoint({
            'epoch': args.epochs,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best=False, filename=filename)

        print(f"Model checkpoint and metadata has been saved at {log_dir}.")
        print("Completed self-supervised training.")

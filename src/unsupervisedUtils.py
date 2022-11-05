import torch 
import logging 
import numpy as np  
import h5py
import time 
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt  
import os   
import yaml 
import shutil 
import wandb 

def get_batch(args,it):
    if args.n_views == 2:
        (image1, image2), text, mask, label = next(it)
        batch = {
                'image1': image1,
                'image2': image2,
                'text': text,
                'mask': mask,
                'label': label
        }
    else:
        image, text, mask, label = next(it)
        batch = {
                'image1': image,
                'image2': None,
                'text': text,
                'mask': mask,
                'label': label
        }
    batch = {k:v.to(args.device) if v is not None else None for k, v in batch.items()}
    return batch

def save_embed(args,model, train_loader,log_dir):
    model.eval()
    # save_config_file(self.writer.log_dir, self.args)

    print(f"Start Embedding Visualization.")
    print(f"Using args: {args}")

    trainiterator = iter(train_loader)
    image_embeddings = []
    text_embeddings = []
    all_labels = []
    with torch.no_grad():
        for loader_idx in range(len(train_loader)):
            batch = get_batch(args,trainiterator)
            im = model.encode_image(batch['image'])
            txt = model.encode_text(batch['text'])
            image_embeddings.append(im.cpu().numpy())
            text_embeddings.append(txt.cpu().numpy())
            all_labels.append(batch['label'].cpu().numpy())
            print("Runs")

    classes= {1: 'Hateful', 0: 'Non-hateful'}
    image_embeddings = np.concatenate(image_embeddings, axis=0)
    text_embeddings = np.concatenate(text_embeddings, axis=0)
    all_labels = np.concatenate(all_labels)
    class_labels = [classes[i.item()] for i in all_labels]
    print("Image Embeddings:", image_embeddings.shape)
    print("Text Embeddings:", text_embeddings.shape)
    print("Labels:", all_labels.shape)

    h_file = h5py.File(os.path.join(log_dir,'embeds.h5'), 'w')
    h_file.create_dataset('image_embeds', data=image_embeddings)
    h_file.create_dataset('text_embeds', data=text_embeddings)
    h_file.create_dataset('labels_embeds', data=all_labels)
    h_file.close()

    green = all_labels == 0
    red = all_labels == 1

    for perplexity in [5, 25, 50, 100]:
        t1 = time.time()
        tsne = TSNE(2, perplexity, n_iter=2500, n_jobs=12)
        Xi_embed = tsne.fit_transform(image_embeddings)
        t2 = time.time()
        print(f"TSNE {perplexity} took {t2-t1}s on image_embeddings")
        t1 = time.time()
        tsne = TSNE(2, perplexity, n_iter=2500, n_jobs=12)
        Xt_embed = tsne.fit_transform(text_embeddings)
        t2 = time.time()
        print(f"TSNE {perplexity} took {t2-t1}s on text_embeddings")
        h_file = h5py.File(os.path.join(log_dir,'embeds-tsne-{}.h5'.format(perplexity)), 'w')
        h_file.create_dataset('image_embeds', data=Xi_embed)
        h_file.create_dataset('text_embeds', data=Xt_embed)
        h_file.create_dataset('labels_embeds', data=all_labels)
        h_file.close()

        plt.figure(figsize=(15, 8))
        plt.title(f"TSNE {perplexity} Text Embeddings")
        plt.scatter(Xt_embed[green, 0], Xt_embed[green, 1], c='g')
        plt.scatter(Xt_embed[red, 0], Xt_embed[red, 1], c='r')
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir,'embeds-tsne-text-{}.png'.format(perplexity)))
        plt.cla()

        plt.figure(figsize=(15, 8))
        plt.title(f"TSNE {perplexity} Image Embeddings")
        plt.scatter(Xi_embed[green, 0], Xi_embed[green, 1], c='g')
        plt.scatter(Xi_embed[red, 0], Xi_embed[red, 1], c='r')
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir,'embeds-tsne-image-{}.png'.format(perplexity)))
        plt.cla()
        print(f"TSNE {perplexity} plot saved")
    print("Complete")

def save_config_file(model_checkpoints_folder, args):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
            yaml.dump(args, outfile, default_flow_style=False)

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def compute_loss(args,out,epoch,meme_mmloss=None,meme_floss=None,mmcontr_loss=None):
    loss = 0.0
    if args.mmcontr:
        immloss, tmmloss = mmcontr_loss(out['image'], out['text'])
        loss += immloss + tmmloss
        wandb.log({'Unsupervised/ImageMMContrLoss': immloss.item()})
        wandb.log({'Unsupervised/TextMMContrLoss': tmmloss.item()})
    if args.memeloss:
        f2i, i2f = meme_mmloss(out['fusion'], out['image'])
        fusion2i = f2i * 0.7 + 0.3 * i2f
        f2t, t2f = meme_mmloss(out['fusion'], out['text'])
        fusion2t = f2t * 0.7 + 0.3 * t2f
        f2f = meme_floss(out['fusion1'], out['fusion2'])
        loss += args.w_f2i * fusion2i + args.w_f2t * fusion2t + args.w_f2f * f2f
        wandb.log({"Unsupervised/Fusion2ImageLoss":fusion2i.item(),"epoch":epoch})
        wandb.log({"Unsupervised/Fusion2TextLoss":fusion2t.item(),"epoch":epoch})
        wandb.log({"Unsupervised/Fusion2FusionLoss":f2f.item(),"epoch":epoch})
    return loss
        # writer.add_scalar('Unsupervised/Fusion2ImageLoss', fusion2i.item(), step)
        # writer.add_scalar('Unsupervised/Fusion2TextLoss', fusion2t.item(), step)
        # writer.add_scalar('Unsupervised/Fusion2FusionLoss', f2f.item(), step)

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
        patience (int): How long to wait after last time validation loss improved.
            Default: 7
        verbose (bool): If True, prints a message for each validation loss improvement.
            Default: False
        delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
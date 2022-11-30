from functools import partial
from torch.multiprocessing import cpu_count
from torchvision.datasets import STL10
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, TensorDataset
from torch import Tensor
import torchvision.transforms as T
import pandas as pd
from torch.utils.data import Dataset
import os
from glob import glob
# from torchvision.io import read_image
from PIL import Image
import json
import torch
from transformers import RobertaTokenizer, DistilBertTokenizer
import numpy as np
from tqdm import tqdm
import clip 
import open_clip
from .preprocess_text import preprocess
# import nlpaug.augmenter.char as nac
# import nlpaug.augmenter.word as naw
# from random import random, randint

import re
import nltk
import heapq
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

from . import data_utils
import os
# import data_utils

os.environ["TOKENIZERS_PARALLELISM"] = "false"

 
class DefaultImgTransform:

    def __init__(self, img_size):
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.train_transform = T.Compose(
           [
            T.Resize((img_size,img_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                    std=[0.26862954, 0.26130258, 0.27577711])
        ]
        )

        self.test_transform = T.Compose(
            [
            T.Resize((img_size,img_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                    std=[0.26862954, 0.26130258, 0.27577711])
        ]
        )

class TextTransform():
    def __init__(self):
        pass 
    def precompute_stats(self, texts, path_save_vocab='data/vocab.json'):
        pass
    def __call__(self, raw_texts):
        texts = []
        for t in raw_texts:
            texts.append(preprocess(t))
        outputs = {}
        toks = clip.tokenize(texts,truncate=True)
        outputs["clip_tokens"] = toks
        return outputs

class ImageText(Dataset):
    def __init__(self, img_folder, metadata_csv, is_labeled, img_col='file_name', 
    text_col='Text Transcription', label_cols='misogynous', im_transforms=None,txt_transform=None, target_transform=None):

        self.metadata = pd.read_csv(metadata_csv)
        self.is_labeled = is_labeled
        if self.is_labeled:
            assert not label_cols is None, "supervised data needs labels"
            self.labels = self.metadata[label_cols].values # can extract from both series and DataFrame

        self.img_names = self.metadata[img_col].to_list()
        self.img_folder = img_folder
        self.img_paths = [os.path.join(img_folder, img_name) for img_name in self.img_names]

        self.texts = self.metadata[text_col].to_list()

        self.num_samples = len(self.metadata)

        self.image_transform = im_transforms
        # self.text_transform = txt_transform
        self.target_transform = target_transform

        if not self.target_transform is None:
            self.labels = self.target_transform(self.labels) 

        if not txt_transform is None:
            self.text_transform_outputs = txt_transform(self.texts)
        else:
            self.text_transform_outputs = None

        print("Loaded and processed {} Samples in {}".format(self.num_samples, metadata_csv))
        
        

    def __getitem__(self, index):
        imgpath = self.img_paths[index]
        img = Image.open(imgpath).convert('RGB')
        if not self.image_transform is None:
            img = self.image_transform(img)

        if not self.text_transform_outputs is None:
            text = {key: values[index] for key, values in self.text_transform_outputs.items()}
        else:
            text = self.texts[index]
        
        if self.is_labeled:
            label = self.labels[index]
            return (img, text), label
        else:                
            return img, text

    def __len__(self):
        return self.num_samples


def create_semi_supervised_dataloaders(args, train_img_dir, train_labeled_csv, train_unlabeled_csv, 
                                                val_img_dir, val_csv, batch_size, image_size, inbatch_label_ratio=None, debug=False,input_resolution=None):
    # args.use_augmentation = False
    label_cols = ['misogynous']

    if args.use_clip:
        image_size = input_resolution

    im_transforms = DefaultImgTransform(img_size=image_size)

    txt_transforms = TextTransform()
        
    if inbatch_label_ratio is None:
        inbatch_label_ratio = data_utils.calculate_label_ratio(train_labeled_csv, train_unlabeled_csv)

    train_sup = ImageText(train_img_dir, metadata_csv=train_labeled_csv, is_labeled=True,
                                im_transforms=im_transforms.test_transform, txt_transform=txt_transforms, label_cols=label_cols)
    train_supervised_loader = DataLoader(dataset=train_sup, batch_size=int(inbatch_label_ratio*batch_size), 
                                            num_workers=cpu_count()//2, drop_last=True, shuffle=True)
    train_unsup = ImageText(train_img_dir, metadata_csv=train_unlabeled_csv, is_labeled=False,
                                im_transforms=im_transforms.test_transform, 
                                txt_transform=txt_transforms, label_cols=label_cols)
    train_unsupervised_loader = DataLoader(dataset=train_unsup, batch_size=batch_size - int(inbatch_label_ratio*batch_size),
                                            num_workers=cpu_count()//2, drop_last=True, shuffle=True)

    val = ImageText(val_img_dir, metadata_csv=val_csv, is_labeled=True,
                            im_transforms=im_transforms.test_transform, txt_transform=txt_transforms, label_cols=label_cols)

    # collate_fn_batch = ...
    # dataloader = DataLoader(dataset=data, collate_fn = collate_fn_batch, batch_size=batch_size, num_workers=cpu_count()//2, drop_last=True, shuffle=True)
    val_loader = DataLoader(dataset=val, batch_size=batch_size, num_workers=cpu_count()*3//4, drop_last=False, shuffle=False)
    if not debug:
        return train_supervised_loader, train_unsupervised_loader, val_loader
    else:
        return train_supervised_loader, train_unsupervised_loader, val_loader, im_transforms, txt_transforms

def create_semi_supervised_test_dataloaders(args, test_img_dir, test_csv, batch_size, image_size, debug=False,input_resolution=None):
    label_cols = ['shaming', 'stereotype', 'objectification', 'violence']
    
    if args.use_clip:
        image_size = input_resolution

    im_transforms = DefaultImgTransform(img_size=image_size)
    txt_transforms = TextTransform()
        
    test = ImageText(test_img_dir, metadata_csv=test_csv, is_labeled=True,
                            im_transforms=im_transforms.test_transform, txt_transform=txt_transforms, label_cols=label_cols)

    test_loader = DataLoader(dataset=test, batch_size=batch_size, num_workers=cpu_count()*3//4, drop_last=False, shuffle=False)

    if not debug:
        return test_loader
    else:
        return test_loader, im_transforms, txt_transforms

def create_dataloader_clip_extractor(args, input_img_dir_clip_extractor, input_file_clip_extractor, batch_size, image_size, debug=False):
    im_transforms = DefaultImgTransform(img_size=image_size) 
    txt_transforms = TextTransform()
    data = ImageText(input_img_dir_clip_extractor, 
                     metadata_csv=input_file_clip_extractor, is_labeled=False,
                     im_transforms=im_transforms.test_transform, txt_transform=txt_transforms)
    
    loader = DataLoader(dataset=data, batch_size=batch_size,
                         num_workers=os.cpu_count()*3//4, drop_last=False, shuffle=False)

    if not debug:
        return loader
    else:
        return loader, im_transforms, txt_transforms

def create_dataloader_pre_extracted(args, image_features_path, text_features_path, batch_size, is_labeled=False, label_path=None, label_cols=None, shuffle=False, normalize=False, feature_stats=None):
    image_features_arr = np.loadtxt(image_features_path)
    text_features_arr = np.loadtxt(text_features_path)
    
    if normalize:
        assert not feature_stats is None, "to normalize, need feature stats"
        image_features_arr = (image_features_arr-feature_stats['image_features_mean'])/feature_stats['image_features_std']
        text_features_arr = (text_features_arr-feature_stats['text_features_mean'])/feature_stats['text_features_std']
        
    if is_labeled:
        assert (not label_cols is None) and (not label_path is None), "supervised data needs labels"
        metadata = pd.read_csv(label_path)
        # can extract from both series and DataFrame
        labels = metadata[label_cols].values
        data = TensorDataset(Tensor(image_features_arr), Tensor(text_features_arr), Tensor(labels))
    else:
        data = TensorDataset(Tensor(image_features_arr), Tensor(text_features_arr))
        
    loader = DataLoader(data, shuffle=shuffle, batch_size=batch_size)

    return loader
# from ..arguments import get_args

# def main():
#     args = get_args()

#     train_supervised_loader, train_unsupervised_loader, val_loader = create_semi_supervised_dataloaders(args,
#                                                                                                         train_img_dir='data/MAMI_processed/images/train',
#                                                                                                         train_labeled_csv='data/MAMI_processed/train_labeled_ratio-0.3.csv',
#                                                                                                         train_unlabeled_csv='data/MAMI_processed/train_unlabeled_ratio-0.3.csv',
#                                                                                                         val_img_dir = 'data/MAMI_processed/images/val',
#                                                                                                         val_csv='data/MAMI_processed/val.csv',
#                                                                                                         batch_size=64, image_size=384)

#     next(iter(train_supervised_loader))
#     next(iter(train_unsupervised_loader))
#     next(iter(val_loader))

#     return 0

# if __name__ == "__main__":
#     main()
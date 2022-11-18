from functools import partial
from torch.multiprocessing import cpu_count
from torchvision.datasets import STL10
from torch.utils.data import DataLoader
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


class ImgAugment:

    def __init__(self, img_size, s=1):
        color_jitter = T.ColorJitter(
            0.2 * s, 0.2 * s, 0.2 * s
        )
        # 10% of the image
        blur = T.GaussianBlur((3, 3), (0.1, 2.0))

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.train_transform = T.Compose(
            [
            # T.RandomResizedCrop(size=img_size, scale=(0.8, 1.0), ratio=(0.8, 1.2)),
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(p=0.5),  # with 0.5 probability
            T.RandomApply([color_jitter], p=0.5),
            T.RandomApply([blur], p=0.5),
            # T.RandomGrayscale(p=0.2),
            T.ToTensor(),
            T.Normalize(mean=self.mean, std=self.std)
            ]
        )

        self.test_transform = T.Compose(
            [
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=self.mean, std=self.std)
            ]
        )

class DefaultImgTransform:

    def __init__(self, img_size):
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.train_transform = T.Compose(
            [
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=self.mean, std=self.std)
            ]
        )

        self.test_transform = T.Compose(
            [   
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=self.mean, std=self.std)
            ]
        )

class TextTransform():
    def __init__(self, txt_bert_model=None, txt_bert_max_length=128, use_sbert=False, vocab_path=None):
        self.vocab_path = vocab_path
        if vocab_path is None:
            self.sentence_vectors_extractor = CountVectorizer(max_features=3000, binary=True)
        else:
            self.vocab = json.load(open(vocab_path))
            self.sentence_vectors_extractor = CountVectorizer(max_features=3000, binary=True, vocabulary=self.vocab)

        self.use_sbert = use_sbert
        if use_sbert:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')

        self.txt_bert_model = txt_bert_model
        if not txt_bert_model is None:
            if txt_bert_model == 'roberta-base':
                self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
            else:
                self.tokenizer = DistilBertTokenizer.from_pretrained(txt_bert_model, model_max_length=txt_bert_max_length)

    def precompute_stats(self, texts, path_save_vocab='data/vocab.json'):
        self.sentence_vectors_extractor.fit(texts)

        # save vocabulary
        self.vocab = self.sentence_vectors_extractor.vocabulary_
        self.vocab_path = path_save_vocab

        # value from vectorizer is numpy type, convert to int to serialize in json
        json.dump({key: int(value) for (key, value) in self.vocab.items()}, open(path_save_vocab, 'w'))
        print(f"Compute vocabulary for {len(texts)} sentences. Save to {path_save_vocab}")

    def __call__(self, texts):
        outputs = dict()
        sentence_vectors = self.sentence_vectors_extractor.transform(texts).toarray()
        outputs['sentence_vectors'] = sentence_vectors

        if self.use_sbert:
            sbert_embedding = self.model.encode(texts)
            outputs['sbert_embedding'] = sbert_embedding

        if not self.txt_bert_model is None:
            toks = self.tokenizer(texts, padding='max_length', truncation=True, return_tensors="pt")
            # output of bert tokenizer is a dictionary
            outputs.update(toks)
        
        return outputs

class ImageText(Dataset):
    def __init__(self, img_folder, metadata_csv, is_labeled, img_col='file_name', text_col='Text Transcription', label_cols='misogynous', im_transforms=None, txt_transform=None, target_transform=None):
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
                                                val_img_dir, val_csv, batch_size, image_size, inbatch_label_ratio=None, debug=False):
    # args.use_augmentation = False
    label_cols = ['shaming', 'stereotype', 'objectification', 'violence']

    if args.use_augmentation:
        im_transforms = ImgAugment(img_size=image_size)
    else:
        im_transforms = DefaultImgTransform(img_size=image_size)

    txt_transforms = TextTransform(use_sbert=False, txt_bert_model=None)
    target_transforms = None

    # need compute vocab before transform text
    txt_transforms.precompute_stats(data_utils.get_texts(train_labeled_csv, train_unlabeled_csv, text_col=args.text_col))

    train_sup = ImageText(train_img_dir, metadata_csv=train_labeled_csv, is_labeled=True,
                            im_transforms=im_transforms.train_transform, txt_transform=txt_transforms, label_cols=label_cols)

    train_unsup = ImageText(train_img_dir, metadata_csv=train_unlabeled_csv, is_labeled=False,
                            im_transforms=im_transforms.train_transform, txt_transform=txt_transforms, label_cols=label_cols)

    val = ImageText(val_img_dir, metadata_csv=val_csv, is_labeled=True,
                            im_transforms=im_transforms.test_transform, txt_transform=txt_transforms, label_cols=label_cols)

    # collate_fn_batch = ...
    # dataloader = DataLoader(dataset=data, collate_fn = collate_fn_batch, batch_size=batch_size, num_workers=cpu_count()//2, drop_last=True, shuffle=True)
    
    if inbatch_label_ratio is None:
        inbatch_label_ratio = data_utils.calculate_label_ratio(train_labeled_csv, train_unlabeled_csv)

    train_supervised_loader = DataLoader(dataset=train_sup, batch_size=int(inbatch_label_ratio*batch_size), 
                                            num_workers=cpu_count()//2, drop_last=True, shuffle=True)
    train_unsupervised_loader = DataLoader(dataset=train_unsup, batch_size=batch_size - int(inbatch_label_ratio*batch_size),
                                            num_workers=cpu_count()//2, drop_last=True, shuffle=True)
    
    val_loader = DataLoader(dataset=val, batch_size=batch_size, num_workers=cpu_count()*3//4, drop_last=False, shuffle=False)

    if not debug:
        return train_supervised_loader, train_unsupervised_loader, val_loader
    else:
        return train_supervised_loader, train_unsupervised_loader, val_loader, im_transforms, txt_transforms

def create_semi_supervised_test_dataloaders(args, test_img_dir, test_csv, batch_size, image_size, debug=False):
    label_cols = ['shaming', 'stereotype', 'objectification', 'violence']

    im_transforms = DefaultImgTransform(img_size=image_size)
    txt_transforms = TextTransform(use_sbert=True, txt_bert_model='distilbert-base-uncased', vocab_path='data/vocab.json')
    target_transforms = None

    test = ImageText(test_img_dir, metadata_csv=test_csv, is_labeled=True,
                            im_transforms=im_transforms.test_transform, txt_transform=txt_transforms, label_cols=label_cols)

    test_loader = DataLoader(dataset=test, batch_size=batch_size, num_workers=cpu_count()*3//4, drop_last=False, shuffle=False)

    if not debug:
        return test_loader
    else:
        return test_loader, im_transforms, txt_transforms


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
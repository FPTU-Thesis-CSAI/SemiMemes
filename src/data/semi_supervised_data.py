from functools import partial
from torch.multiprocessing import cpu_count
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

from transformers import LxmertTokenizer

from . import data_utils
import os
# import data_utils

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class ImgAugment:

    def __init__(self, img_size, s=1):

        self.mean = [0.48145466, 0.4578275, 0.40821073]
        self.std = [0.26862954, 0.26130258, 0.27577711]
        # color_jitter = T.ColorJitter(
        #     0.2 * s, 0.2 * s, 0.2 * s
        # )
        # # 10% of the image
        # blur = T.GaussianBlur((3, 3), (0.1, 2.0))


        self.train_transform = T.Compose(
          [
            T.RandomResizedCrop(img_size, interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(),
            # transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.2),
            T.RandomGrayscale(p=0.1),
            # transforms.RandomPerspective(),
            T.RandomRotation(degrees=10),
            T.ToTensor(),
            T.Normalize(mean=self.mean,std=self.std)
    ]
        )

        self.test_transform = T.Compose([
            T.Resize((img_size,img_size), interpolation=T.InterpolationMode.BICUBIC),
            # transforms.CenterCrop(clip_model.visual.input_resolution),
            T.ToTensor(),
            T.Normalize(mean=self.mean,std=self.std)
        ]
            )


class DefaultImgTransform:

    def __init__(self, img_size):
        self.mean = [0.48145466, 0.4578275, 0.40821073]
        self.std = [0.26862954, 0.26130258, 0.27577711]

        self.train_transform = T.Compose(
            [
                T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=self.mean, std=self.std)
            ]
        )

        self.test_transform = T.Compose(
            [
                T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=self.mean, std=self.std)
            ]
        )


class TextTransform():
    def __init__(self,args, txt_bert_model=None, txt_max_length=128, use_sbert=False,
                use_countvectorizer=False,vocab_path=None,use_clip=None):
        self.vocab_path = vocab_path
        self.use_countvectorizer = use_countvectorizer
        if self.use_countvectorizer:
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
                self.tokenizer = RobertaTokenizer.from_pretrained(
                    "roberta-base")
            elif txt_bert_model == 'lxmert_tokenizer':
                self.tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
            else:
                self.tokenizer = DistilBertTokenizer.from_pretrained(txt_bert_model, model_max_length=txt_max_length)
        self.use_clip = use_clip
        self.args = args 
        if args.use_open_clip:
            self.tokenizer = open_clip.get_tokenizer('xlm-roberta-large-ViT-H-14')

    def precompute_stats(self, texts, path_save_vocab='data/vocab.json'):
        self.sentence_vectors_extractor.fit(texts)

        # save vocabulary
        self.vocab = self.sentence_vectors_extractor.vocabulary_
        self.vocab_path = path_save_vocab

        # value from vectorizer is numpy type, convert to int to serialize in json
        json.dump({key: int(value) for (key, value)
                  in self.vocab.items()}, open(path_save_vocab, 'w'))
        print(
            f"Compute vocabulary for {len(texts)} sentences. Save to {path_save_vocab}")

    def __call__(self, raw_texts):
        texts = []
        for t in raw_texts:
            texts.append(preprocess(t))

        outputs = dict()
        if self.use_countvectorizer:
            sentence_vectors = self.sentence_vectors_extractor.transform(texts).toarray()
            outputs['sentence_vectors'] = sentence_vectors

        if self.use_sbert:
            sbert_embedding = self.model.encode(texts)
            outputs['sbert_embedding'] = sbert_embedding

        if not self.txt_bert_model is None:
            toks = self.tokenizer(
                texts, padding='max_length', truncation=True, return_tensors="pt")
            # output of bert tokenizer is a dictionary
            outputs.update(toks)
        
        if self.use_clip:
            if self.args.use_open_clip:
                toks = self.tokenizer(texts)
                outputs["clip_tokens"] = toks
            else:
                toks = clip.tokenize(texts,truncate=True)
                outputs["clip_tokens"] = toks
                
        return outputs


class CaptionTransform():
    def __init__(self, args, txt_bert_model=None, txt_max_length=128, use_sbert=False,
                use_countvectorizer=False,vocab_path=None,use_clip=None):
        # self.vocab_path = vocab_path
        # self.use_countvectorizer = use_countvectorizer
        # if self.use_countvectorizer:
        #     if vocab_path is None:
        #         self.sentence_vectors_extractor = CountVectorizer(max_features=3000, binary=True)
        #     else:
        #         self.vocab = json.load(open(vocab_path))
        #         self.sentence_vectors_extractor = CountVectorizer(max_features=3000, binary=True, vocabulary=self.vocab)

        self.use_sbert = use_sbert
        if use_sbert:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')

        self.txt_bert_model = txt_bert_model
        if not txt_bert_model is None:
            if txt_bert_model == 'roberta-base':
                self.tokenizer = RobertaTokenizer.from_pretrained(
                    "roberta-base")
            elif txt_bert_model == 'lxmert_tokenizer':
                self.tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
            else:
                self.tokenizer = DistilBertTokenizer.from_pretrained(txt_bert_model, model_max_length=txt_max_length)
        
        self.use_clip = use_clip
        self.args = args
        if args.use_open_clip:
            self.tokenizer = open_clip.get_tokenizer('xlm-roberta-large-ViT-H-14')

    # def precompute_stats(self, texts, path_save_vocab='data/vocab.json'):
    #     self.sentence_vectors_extractor.fit(texts)

    #     # save vocabulary
    #     self.vocab = self.sentence_vectors_extractor.vocabulary_
    #     self.vocab_path = path_save_vocab

    #     # value from vectorizer is numpy type, convert to int to serialize in json
    #     json.dump({key: int(value) for (key, value)
    #               in self.vocab.items()}, open(path_save_vocab, 'w'))
    #     print(
    #         f"Compute vocabulary for {len(texts)} sentences. Save to {path_save_vocab}")

    def __call__(self, texts):
        outputs = dict()
        # if self.use_countvectorizer:
        #     sentence_vectors = self.sentence_vectors_extractor.transform(texts).toarray()
        #     outputs['sentence_vectors'] = sentence_vectors

        if self.use_sbert:
            sbert_embedding = self.model.encode(texts)
            outputs['sbert_embedding'] = sbert_embedding

        if not self.txt_bert_model is None:
            toks = self.tokenizer(
                texts, padding='max_length', truncation=True, return_tensors="pt")
            # output of bert tokenizer is a dictionary
            outputs.update(toks)
        
        if self.use_clip:
            if self.args.use_open_clip:
                toks = self.tokenizer(texts)
                outputs["clip_tokens"] = toks
            else:
                toks = clip.tokenize(texts,truncate=True)
                outputs["clip_tokens"] = toks
        return outputs

class ImageText(Dataset):
    def __init__(self, img_folder, metadata_csv, is_labeled, img_col='file_name', text_col='Text Transcription', label_cols='misogynous', im_transforms=None, txt_transform=None, target_transform=None, second_txt_transform=None):
        self.metadata = pd.read_csv(metadata_csv)

        self.is_labeled = is_labeled
        if self.is_labeled:
            assert not label_cols is None, "supervised data needs labels"
            # can extract from both series and DataFrame
            self.labels = self.metadata[label_cols].values

        self.img_names = self.metadata[img_col].to_list()
        self.img_folder = img_folder
        self.img_paths = [os.path.join(img_folder, img_name)
                        for img_name in self.img_names]

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

        if not second_txt_transform is None:
            second_text_col = 'generated_caption'
            self.second_texts = self.metadata[second_text_col]
            self.second_text_transform_outputs = second_txt_transform(self.second_texts)
        else:
            self.second_text_transform_outputs = None

        print("Loaded and processed {} Samples in {}".format(
            self.num_samples, metadata_csv))

    def __getitem__(self, index):
        imgpath = self.img_paths[index]
        img = Image.open(imgpath).convert('RGB')
        if not self.image_transform is None:
            img = self.image_transform(img)

        if not self.text_transform_outputs is None:
            text = {key: values[index]
                    for key, values in self.text_transform_outputs.items()}
        else:
            text = self.texts[index]
            
        if not self.second_text_transform_outputs is None:
            second_text = {'caption_'+key: values[index]
                    for key, values in self.second_text_transform_outputs.items()}
            text.update(second_text)

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
    label_cols = ['shaming', 'stereotype', 'objectification', 'violence']

    if args.use_clip:
        image_size = input_resolution

    if args.use_augmentation:
        im_transforms = ImgAugment(img_size=image_size)
    else:
        im_transforms = DefaultImgTransform(img_size=image_size)
    if not args.use_bert_model:
        args.pretrain_bert_model = None
    txt_transforms = TextTransform(args,use_sbert=args.use_bert_embedding,
        txt_bert_model=args.pretrain_bert_model,
        use_countvectorizer=args.use_sentence_vectorizer,
        use_clip=args.use_clip)

    # caption_transform = CaptionTransform(args, use_sbert=args.use_bert_embedding,
    #     txt_bert_model=args.pretrain_bert_model,
    #     use_countvectorizer=args.use_sentence_vectorizer,
    #     use_clip=args.use_clip)
    
    caption_transform = None
    
    # if args.dual_stream:
    #     txt_transforms = TextTransform(
    #         use_sbert=True, txt_bert_model='lxmert_tokenizer')
    # else:
    #     txt_transforms = TextTransform(
    #         use_sbert=True, txt_bert_model='distilbert-base-uncased')
    target_transforms = None

    # need compute vocab before transform text
    if args.use_sentence_vectorizer:
        txt_transforms.precompute_stats(data_utils.get_texts(
            train_labeled_csv, train_unlabeled_csv, text_col=args.text_col))

    train_sup = ImageText(train_img_dir, metadata_csv=train_labeled_csv, is_labeled=True,
                        im_transforms=im_transforms.test_transform, txt_transform=txt_transforms, label_cols=label_cols, second_txt_transform=caption_transform)

    train_unsup = ImageText(train_img_dir, metadata_csv=train_unlabeled_csv, is_labeled=False,
                            im_transforms=im_transforms.test_transform, txt_transform=txt_transforms, label_cols=label_cols, second_txt_transform=caption_transform)

    val = ImageText(val_img_dir, metadata_csv=val_csv, is_labeled=True,
                    im_transforms=im_transforms.test_transform, txt_transform=txt_transforms, label_cols=label_cols, second_txt_transform=caption_transform)

    # collate_fn_batch = ...
    # dataloader = DataLoader(dataset=data, collate_fn = collate_fn_batch, batch_size=batch_size, num_workers=cpu_count()//2, drop_last=True, shuffle=True)

    if inbatch_label_ratio is None:
        inbatch_label_ratio = data_utils.calculate_label_ratio(
            train_labeled_csv, train_unlabeled_csv)

    train_supervised_loader = DataLoader(dataset=train_sup, batch_size=int(inbatch_label_ratio*batch_size),
                                        num_workers=cpu_count()//2, drop_last=True, shuffle=True)
    train_unsupervised_loader = DataLoader(dataset=train_unsup, batch_size=batch_size - int(inbatch_label_ratio*batch_size),
                                        num_workers=cpu_count()//2, drop_last=True, shuffle=True)

    val_loader = DataLoader(dataset=val, batch_size=batch_size,
                            num_workers=cpu_count()*3//4, drop_last=False, shuffle=False)

    if not debug:
        return train_supervised_loader, train_unsupervised_loader, val_loader
    else:
        return train_supervised_loader, train_unsupervised_loader, val_loader, im_transforms, txt_transforms

def create_semi_supervised_test_dataloaders(args, test_img_dir, test_csv, batch_size, image_size, debug=False,input_resolution=None):
    label_cols = ['shaming', 'stereotype', 'objectification', 'violence']
    
    if args.use_clip:
        image_size = input_resolution

    if args.use_augmentation:
        im_transforms = ImgAugment(img_size=image_size)
    else:
        im_transforms = DefaultImgTransform(img_size=image_size)
        
    txt_transforms = TextTransform(args, use_sbert=args.use_bert_embedding,
        txt_bert_model=args.pretrain_bert_model,
        use_countvectorizer=args.use_sentence_vectorizer,
        use_clip=args.use_clip, vocab_path='data/vocab.json')
    
    caption_transform = CaptionTransform(args, use_sbert=args.use_bert_embedding,
        txt_bert_model=args.pretrain_bert_model,
        use_countvectorizer=args.use_sentence_vectorizer,
        use_clip=args.use_clip)
    
    caption_transform = None
        
    target_transforms = None

    test = ImageText(test_img_dir, metadata_csv=test_csv, is_labeled=True,
                    im_transforms=im_transforms.test_transform, txt_transform=txt_transforms, label_cols=label_cols, second_txt_transform=caption_transform)

    test_loader = DataLoader(dataset=test, batch_size=batch_size,
                            num_workers=cpu_count()*3//4, drop_last=False, shuffle=False)

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

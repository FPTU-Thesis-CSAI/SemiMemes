
from cgitb import text
import os
from random import random, randint
import cv2
import glob
import json
from tqdm.auto import tqdm
from PIL import Image, ImageFile
from data.utils import stratified_sample_df
ImageFile.LOAD_TRUNCATED_IMAGES = True 
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import pandas as pd 
import numpy as np
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
from torchvision.transforms import transforms
from .gaussian_blur import GaussianBlur 
from transformers import RobertaTokenizer,DistilBertTokenizer
from .view_generator import ContrastiveLearningViewGenerator 
from torch.utils.data import ConcatDataset

class ImageTextClassificationForVLMDataset(Dataset):
    def __init__(self, img_feature_path, csv_path, supervise = True,model_type="visualbert", vilt_processor=None,mode='train',superviseunsuperviseproportion = [3, 7], debug=False, metadata_path = None, augmentation=False): 
        self.supervise = supervise
        self.model_type = model_type
        self.train=False
        self.pro = superviseunsuperviseproportion

        self.augmentation = augmentation
        if self.augmentation:
            self.augocr = nac.OcrAug(aug_char_max=5)
            self.augkey = nac.KeyboardAug(aug_char_max=5)

            self.augspell = naw.SpellingAug(aug_max=5)
            self.augswap = naw.RandomWordAug(action="swap", aug_max=5)
            self.augdel = naw.RandomWordAug(aug_max=5)
            self.augsplit = naw.SplitAug(aug_max=5)

            self.list_aug_text = [self.augocr, self.augkey, self.augspell, self.augswap, self.augdel, self.augsplit]

        if mode=='train':
            self.train = True
            img_feature_path += "train_images/"
        elif mode=="val":
            img_feature_path += "val_images/"
        else:
            img_feature_path += "test_images/"

        if self.model_type in ["visualbert", "lxmert"]:
            self.img_features = torch.load(os.path.join(img_feature_path, "features.pt"))
            with open(os.path.join(img_feature_path, "names.txt"), "r") as f:
                lines = f.readlines()
            self.img_names = [line.strip() for line in lines]
            self.img_name2index = {}
            for i, name in enumerate(self.img_names):
                self.img_name2index[name] = i # the i-th vector in img_features
        elif self.model_type == "vilt":
            self.imgs = {}
            img_paths = glob.glob(img_feature_path+"/*.jpg")
            print (f"load images...")
            for img_path in tqdm(img_paths):
                img_name = img_path.split("/")[-1]
                #tmp = Image.open(img_path)
                #keep = tmp.copy()
                #tmp.close()
                self.imgs[img_name] = cv2.imread(img_path)

        if self.model_type == "lxmert":
            self.boxes = torch.load(os.path.join(img_feature_path, "boxes.pt"))

        self.vilt_processor = vilt_processor
        if self.supervise == False:
            labeled_id_path = os.path.join("/".join(csv_path.split("/")[:-1]),"labeled_sample.csv")
            unlabeled_id_path = os.path.join("/".join(csv_path.split("/")[:-1]),"unlabeled_sample.csv")
            self.labeled_ids = pd.read_csv(labeled_id_path)["labeled_id"].tolist()
            self.unlabeled_ids = pd.read_csv(unlabeled_id_path)["unlabeled_id"].tolist()

        self.data_csv = []
        df = pd.read_csv(csv_path)
        for _,row in df.iterrows():
            data_item = {}
            data_item["image"] = str(row["Id"])+".jpg"
            data_item["caption"] = row["ocr_text"]
            labels = []
            if mode != "test":
                if row["humour"] == "not_funny":
                    labels.append(0)
                else:
                    labels.append(1)
                if row["sarcastic"] == "not_sarcastic":
                    labels.append(0)
                else:
                    labels.append(1)
                if row["offensive"] == "not_offensive":
                    labels.append(0)
                else:
                    labels.append(1)
                if row["motivational"] == "not_motivational":
                    labels.append(0)
                else:
                    labels.append(1)
                data_item["labels"] = labels
            self.data_csv.append(data_item)

        self.debug = debug
        if self.debug:
            assert not metadata_path is None, "provide metadata to debug"
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
    
    def augment_features(self, img_features, rate=0.75):
        sample_features_idx = np.random.choice(len(img_features), size=int(len(img_features)*rate))
        img_features = img_features[sample_features_idx]
        return img_features, sample_features_idx

    def augment_text(self, text):
        
        if random() < 0.5:
            i = randint(0, len(self.list_aug_text)-1)
            text = self.list_aug_text[i].augment(text)[0]

        return text


    def __getitem__(self, idx):
        if (self.train==True and self.supervise==True) or self.train==False:
            data_point = self.data_csv[idx]
            if self.model_type == "visualbert":
                img_index = self.img_name2index[data_point["image"]]
                img_features = self.img_features[img_index]
                caption = data_point["caption"]

                if self.augmentation:
                    img_features, idx_features = self.augment_features(img_features, rate=0.75)
                    if len(caption) > 20:
                        caption = self.augment_text(caption)

                if not self.debug:
                    return caption, img_features, data_point["labels"]
                else:
                    try:
                        metadata = self.metadata[img_index]
                    except IndexError as e:
                        print(e)
                        print(f'at index {img_index}')
                        print()
                    return caption, img_features, data_point["labels"], metadata, idx

            elif self.model_type == "lxmert":
                img_index = self.img_name2index[data_point["image"]]
                return data_point["caption"], self.boxes[img_index], self.img_features[img_index], data_point["labels"]
            elif self.model_type == "vilt":
                """
                try:
                    inputs = self.vilt_processor(images=self.imgs[data_point["image"]], text=data_point["caption"], 
                            max_length=32, return_tensors="pt", padding='max_length', truncation=True, 
                            add_special_tokens=True)
                except:
                    print (data_point)
                    exit()
                """
                return self.imgs[data_point["image"]], data_point["caption"], data_point["label"]
        if self.train==True and self.supervise==False:
            supervise_img = []
            supervise_text = []
            supervise_label = []
            supervise_box = []
            for i in range(idx*self.pro[0],(idx+1)*self.pro[0]):
                data_point = self.data_csv[self.labeled_ids[i]-1]
                img_index = self.img_name2index[data_point["image"]]
                supervise_img.append(self.img_features[img_index])
                supervise_text.append(data_point["caption"])
                supervise_label.append(data_point["labels"])
                if self.model_type == "lxmert":
                    supervise_box.append(self.boxes[img_index])
            unsupervise_img = []
            unsupervise_text = []
            unsupervise_label = []
            unsupervise_box = []
            for i in range(idx*self.pro[1],(idx+1)*self.pro[1]):
                data_point = self.data_csv[self.unlabeled_ids[i]-1]
                img_index = self.img_name2index[data_point["image"]]
                unsupervise_img.append(self.img_features[img_index])
                unsupervise_text.append(data_point["caption"])
                unsupervise_label.append(data_point["labels"])
                if self.model_type == "lxmert":
                    unsupervise_box.append(self.boxes[img_index])

            feature = []
            feature.append(supervise_img)
            feature.append(supervise_text)
            feature.append(supervise_box)
            feature.append(unsupervise_img)
            feature.append(unsupervise_text)
            feature.append(unsupervise_box)
            return feature,supervise_label

    def __len__(self):
        if (self.train==True and self.supervise==True) or self.train==False:
            return len(self.data_csv)
        elif self.train==True and self.supervise==False:
            return int(len(self.unlabeled_ids)/self.pro[1])

def np_random_sample(arr, size=1):
        return arr[np.random.choice(len(arr), size=size, replace=False)]

class SupervisionHatefulMemesDataset(Dataset):
    def __init__(self, folder_path, phase, txt_model, height=224, width=224, im_transforms=None, num_samples=0, txt_max_length=100):
        jsonpath = folder_path + '/' + phase + '.jsonl'
        print(jsonpath)
        data = pd.read_json(jsonpath, lines=True)
        if num_samples != 0:
            data = stratified_sample_df(data, 'label', num_samples) 
        self.data = data
        self.num_samples = len(self.data)
        self.phase = phase
        self.labels = np.asarray(self.data['label'])
        self.height = height
        self.width = width
        self.folder_path = folder_path
        self.txt_model = txt_model
        if txt_model == 'roberta-base':
            self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        else:
            self.tokenizer = DistilBertTokenizer.from_pretrained(txt_model, model_max_length=txt_max_length)
        self.encoded_captions = self.tokenizer(list(self.data.text), padding='max_length', truncation=True)
        self.image_transform = im_transforms
        print("Loaded {} Samples in {} Hateful Memes Dataset".format(self.num_samples, phase))
        print(self.data['label'].value_counts())

    def __getitem__(self, index):
        single_image_label = self.labels[index]
        imgpath = self.folder_path + '/' + self.data.img[index]
        img_as_img = Image.open(imgpath).convert('RGB')
        img_views = self.image_transform(img_as_img)

        encoded_caption = {
            key: torch.tensor(values[index])
            for key, values in self.encoded_captions.items()
        }
        text2tokens, att_mask = encoded_caption['input_ids'], encoded_caption['attention_mask']
        return img_views, text2tokens, att_mask, single_image_label

    def __len__(self):
        return self.num_samples 

class SupervisionHarmemesDataset(Dataset):
    def __init__(self, folder_path, phase, txt_model, height=224, width=224, im_transforms=None, num_samples=0, txt_max_length=100):
        jsonpath = folder_path + '/defaults/annotations/' + phase + '.jsonl'
        self.image_dir = folder_path + '/defaults/images/'
        self.data = pd.read_json(jsonpath, lines=True)
        self.phase = phase

        def label_mapping(x):
            for t in x:
                if 'not' in t:
                    return 0
            return 1
        self.data['target'] = self.data['labels'].apply(label_mapping)
        if num_samples != 0:
            self.data = stratified_sample_df(self.data, 'target', num_samples) 
        self.num_samples = len(self.data)
        self.labels = np.asarray(self.data['target'])
        self.height = height
        self.width = width
        self.folder_path = folder_path
        self.txt_model = txt_model
        if txt_model == 'roberta-base':
            self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        else:
            self.tokenizer = DistilBertTokenizer.from_pretrained(txt_model, model_max_length=txt_max_length)
        self.encoded_captions = self.tokenizer(list(self.data.text), padding='max_length', truncation=True)
        self.image_transform = im_transforms
        print("Loaded {} samples in {} Harmemes Dataset".format(self.num_samples, phase))
        print(self.data['target'].value_counts())

    def __getitem__(self, index):
        single_image_label = self.labels[index]
        imgpath = self.image_dir + self.data.image[index]
        img_as_img = Image.open(imgpath).convert('RGB')
        img_views = self.image_transform(img_as_img)

        encoded_caption = {
            key: torch.tensor(values[index])
            for key, values in self.encoded_captions.items()
        }
        text2tokens, att_mask = encoded_caption['input_ids'], encoded_caption['attention_mask']
        return img_views, text2tokens, att_mask, single_image_label

    def __len__(self):
        return self.num_samples

class MMHSDataset(Dataset):
    def __init__(self, folder_path, phase, txt_model, height=224, width=224, im_transforms=None, num_samples=0, txt_max_length=100):
        jsonpath = folder_path  +'/'+ phase + '_ids.txt'
        txtpath = folder_path + '/img_txt/{}.json'
        self.image_dir = folder_path + '/img_resized/'
        with open(jsonpath, 'r') as f:
            ids = f.readlines()
        ids = [t.replace('\n', '') for t in ids]
        if num_samples != 0:
            ids = ids[:num_samples]
        self.phase = phase
        self.text = []
        self.ids = []
        for idx in ids:
            try:
                with open(txtpath.format(idx), 'r') as f:
                    d = json.load(f)
                    self.text.append(d['img_text'])
                    self.ids.append(idx)
            except FileNotFoundError:
                pass
        self.num_samples = len(self.ids)
        print("Loaded {} Samples in {} MMHS Dataset".format(self.num_samples, phase))
        self.height = height
        self.width = width
        self.folder_path = folder_path
        self.txt_model = txt_model
        if txt_model == 'roberta-base':
            self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        else:
            self.tokenizer = DistilBertTokenizer.from_pretrained(txt_model, model_max_length=txt_max_length)
        self.encoded_captions = self.tokenizer(list(self.text), padding='max_length', truncation=True)
        self.image_transform = im_transforms

    def __getitem__(self, index):
        single_image_label = 0  # Not Available 
        imgpath = self.image_dir + str(self.ids[index]) + '.jpg'
        img_as_img = Image.open(imgpath).convert('RGB')
        img_views = self.image_transform(img_as_img)

        encoded_caption = {
            key: torch.tensor(values[index])
            for key, values in self.encoded_captions.items()
        }
        text2tokens, att_mask = encoded_caption['input_ids'], encoded_caption['attention_mask']
        return img_views, text2tokens, att_mask, single_image_label

    def __len__(self):
        return self.num_samples

class MemotionDataset(Dataset):
    def __init__(self, folder_path, phase, task, txt_model, height=224, width=224, im_transforms=None, num_samples=0, txt_max_length=100):
        self.phase = phase
        if self.phase == 'train':
            df_path = os.path.join(folder_path, 'labels.csv')
            self.img_path = os.path.join(folder_path, 'images')
            data = pd.read_csv(df_path)
            try:
                data.drop(columns=['Unnamed: 0'], inplace=True)
            except:
                pass
            if task == 'A' or task == 'a':
                data['label'] = data['overall_sentiment'].apply(self.get_sentiment_label)
            elif task in ['b', 'B']:
                data['label'] = data.apply(lambda x: self.get_task2_label(x), axis=1)
            else:
                data['label'] = data.apply(lambda x: self.get_task3_label(x), axis=1)
        elif self.phase == 'test':
            df_path = os.path.join(folder_path,'memotion_test_data/test_data/2000_testdata.csv') 
            self.img_path = os.path.join(folder_path, 'memotion_test_data/test_data/2000_data')
            text_image_data = pd.read_csv(df_path)
            label_data = pd.read_csv(os.path.join(folder_path,'memotion_test_data/test_data/Meme_groundTruth.csv'))
            data = pd.merge(text_image_data,label_data)
            self.label_dict = {(0,0):0,(0,1):1,(0,2):2,(0,3):3,
                (1,0):4,(1,1):5,(1,2):6,(1,3):7,
                (2,0):8,(2,1):9,(2,2):10,(2,3):11,
                (3,0):12,(3,1):13}
        try:
            data.drop(columns=['Unnamed: 0'], inplace=True)
        except:
            pass

        self.task = task
        if num_samples != 0 and phase =='train':
            if task in ['a', 'A']:
                data = stratified_sample_df(data, 'label', num_samples)
            else:
                data = data.sample(n=num_samples)
        if 'corrected_text' in data:
            data['text_corrected'] = data['corrected_text']
        if 'Image_name' in data:
            data['image_name'] = data['Image_name']
        self.df = data
        self.df['text_corrected'].fillna('None', inplace=True)
        self.num_samples = len(self.df)
        print("Loaded {} Samples in {} Memotion Dataset".format(self.num_samples, phase))
        self.height = height
        self.width = width
        self.phase = phase
        self.folder_path = folder_path
        self.txt_model = txt_model
        if txt_model == 'roberta-base':
            self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        else:
            self.tokenizer = DistilBertTokenizer.from_pretrained(txt_model, model_max_length=txt_max_length)
        self.encoded_captions = self.tokenizer(list(self.df['text_corrected']), padding='max_length', truncation=True)
        self.image_transform = im_transforms

    def get_task3_label(self, row):
        label = [] 
        humour_dict = {
            'not_funny': 0,
            'funny': 1,
            'very_funny': 2,
            'hilarious': 3
        }
        sarcasm_dict = {
            'not_sarcastic': 4,
            'general': 5,
            'twisted_meaning': 6,
            'very_twisted': 7
        }
        offen_dict = {
            'not_offensive': 8,
            'slight': 9,
            'very_offensive': 10,
            'hateful_offensive': 11
        }
        motiv_dict = {
            'not_motivational': 12,
            'motivational': 13
        }

        return [humour_dict[row['humour']], sarcasm_dict[row['sarcasm']], offen_dict[row['offensive']], motiv_dict[row['motivational']]]

    def get_task2_label(self, row):
        label = [] 
        if row['humour'] in ['funny', 'hilarious', 'very_funny']:
            label += [0]
        if row['sarcasm'] in ['general', 'twisted_meaning', 'very_twisted']:
            label += [1]
        if row['motivational'] == 'motivational':
            label += [3]
        if row['offensive'] != 'not_offensive':
            label += [2]
        return label

    def get_sentiment_label(self, label):
        if label in ['positive', 'very_positive']:
            return 0 
        elif label in ['neutral']:
            return 1 
        elif label in ['negative', 'very_negative']:
            return 2 
        return -1

    def __getitem__(self, index):
        if self.task in ['a', 'A']:
            if self.phase == 'test':
                single_image_label = float(self.df['Labels'].iloc[index][0]).split('_')[0]
            else: 
                single_image_label = self.df['label'].iloc[index]
        else:
            if self.phase == "test":
                if self.task in ['b', 'B']:
                    str_label = self.df['Labels'].iloc[index].split('_')[1]
                    str_label = [float(i) for i in str_label]
                    single_image_label = np.array([str_label[0],str_label[1],str_label[2],str_label[3]])
                else: 
                    str_label = self.df['Labels'].iloc[index].split('_')[2]
                    str_label = [float(i) for i in str_label]
                    zeros = np.zeros(14)
                    zeros[self.label_dict[0,str_label[0]]] = 1
                    zeros[self.label_dict[1,str_label[1]]] = 1
                    zeros[self.label_dict[2,str_label[2]]] = 1
                    zeros[self.label_dict[3,str_label[3]]] = 1
            else:
                single_image_label = np.array(self.df['label'].iloc[index], dtype=int)
                zeros = np.zeros(4 if self.task in ['b', 'B'] else 14)
                zeros[single_image_label] = 1
                single_image_label = zeros
        imgpath = os.path.join(self.img_path, self.df.iloc[index]['image_name'])
        img_as_img = Image.open(imgpath)
        if img_as_img.mode is not 'RGBA':
            img_as_img = img_as_img.convert("RGBA")
        img_as_img = img_as_img.convert('RGB')
        img_views = self.image_transform(img_as_img)
        encoded_caption = {
            key: torch.tensor(values[index])
            for key, values in self.encoded_captions.items()
        }
        text2tokens, att_mask = encoded_caption['input_ids'], encoded_caption['attention_mask']
        return img_views, text2tokens, att_mask, single_image_label

    def __len__(self):
        return self.num_samples

def get_std_image_transform(size):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         normalize])
    return data_transforms

def get_simclr_pipeline_transform(size, s=1):
    """Return a set of data augmentation transformations as described in the SimCLR paper."""
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.RandomApply([color_jitter], p=0.8),
                                         transforms.RandomGrayscale(p=0.2),
                                         GaussianBlur(kernel_size=int(0.1 * size)),
                                         transforms.ToTensor(),
                                         normalize])
    return data_transforms


def get_unsupervision_dataset(args):
    num_samples, txt_model = args.n_samples, args.txtmodel
    imtransforms = get_std_image_transform(224) 
    if args.simclr:
        imtransforms = get_simclr_pipeline_transform(224)
    if args.n_views > 1:
        imtransforms = ContrastiveLearningViewGenerator(imtransforms, args.n_views)
    # BEWARE: HARDCODED PATHS FOR MEMES
    path = '/home/viet/SSLMemes/data/hateful_memes'
    dataset1 = SupervisionHatefulMemesDataset(path, 'train', txt_model, im_transforms=imtransforms, num_samples=num_samples)
    path = '/home/viet/SSLMemes/data/MMHS150K'
    dataset2 = MMHSDataset(path, 'train', txt_model, im_transforms=imtransforms, num_samples=num_samples)
    return ConcatDataset([dataset1, dataset2])

def get_supervision_dataset_hateful(args):
    path, txt_model = args.data, args.txtmodel
    num_samples = args.n_samples
    train_transform = get_simclr_pipeline_transform(224)
    val_transform = transforms.Compose([transforms.CenterCrop(size=224),
                                        transforms.ToTensor()])
    traindataset = SupervisionHatefulMemesDataset(
            path, phase='train', txt_model=txt_model, im_transforms=train_transform, num_samples=num_samples)
    valdataset = SupervisionHatefulMemesDataset(path, phase='dev', txt_model=txt_model, im_transforms=val_transform)
    return traindataset, valdataset

def get_supervision_dataset_memotion(args):
    path, txt_model = args.data, args.txtmodel
    num_samples = args.n_samples
    train_transform = get_simclr_pipeline_transform(224)
    val_transform = transforms.Compose([transforms.CenterCrop(size=224),
                                        transforms.ToTensor()])
    traindataset = MemotionDataset(
            path, phase='train', task=args.task, txt_model=txt_model, im_transforms=train_transform, num_samples=num_samples)

    valdataset = MemotionDataset(path, phase='test', task=args.task, txt_model=txt_model, im_transforms=val_transform)
    return traindataset, valdataset

def get_supervision_dataset_harmeme(args):
    path, txt_model = args.data, args.txtmodel
    num_samples = args.n_samples
    train_transform = get_simclr_pipeline_transform(224)
    val_transform = transforms.Compose([transforms.CenterCrop(size=224),
                                        transforms.ToTensor()])
    traindataset = SupervisionHarmemesDataset(
            path, phase='train', txt_model=txt_model, im_transforms=train_transform, num_samples=num_samples)
    valdataset = SupervisionHarmemesDataset(path, phase='test', txt_model=txt_model, im_transforms=val_transform)
    return traindataset, valdataset
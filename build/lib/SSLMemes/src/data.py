
import os
import cv2
import glob
import json
from tqdm.auto import tqdm
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True 
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import pandas as pd 
import wandb 
import numpy as np 
from SSLMemes.feature_extraction.lxmert.processing_image import Preprocess
from SSLMemes.feature_extraction.lxmert.utils import Config

class ImageTextClassificationDataset(Dataset):
    def __init__(self, img_feature_path, csv_path, supervise = True,model_type="visualbert", vilt_processor=None,mode='train',superviseunsuperviseproportion = [3, 7]): 
        self.supervise = supervise
        self.model_type = model_type
        self.train=False
        self.pro = superviseunsuperviseproportion
        self.mode = mode
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
        
            
    def __getitem__(self, idx):
        if (self.train==True and self.supervise==True) or self.train==False:
            data_point = self.data_csv[idx]
            if self.model_type == "visualbert":
                img_index = self.img_name2index[data_point["image"]]
                return data_point["caption"], self.img_features[img_index], data_point["labels"]
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
    def show_image(self, idx=None, data_id=None):
        data_point = self.data_csv[idx]
        labels =  data_point["labels"]
        labels_str = ""
        if int(labels[0]) == 1:
            labels_str += "humour/"
        elif int(labels[0]) == 0:
            labels_str += "not funny/"
        if int(labels[1]) == 1:
            labels_str += "sarcastic/"
        elif int(labels[1]) == 0:
            labels_str += "not sarcastic/"
        if int(labels[2]) == 1:
            labels_str += "offensive/"
        elif int(labels[2]) == 0:
            labels_str += "not offensive/"
        if int(labels[3]) == 1:
            labels_str += "motivational/"
        elif int(labels[3]) == 0:
            labels_str += "not motivational/"

        if self.model_type == "visualbert":

            img_dir = os.path.join("./SSLMemes/data/Memotion2.0/images",f"{self.mode}_images")
            img_path = os.path.join(img_dir,data_point['image'])
            with open(img_path,"rb") as f:
                img = Image.open(f).convert("RGB")
            images = wandb.Image(img, caption=data_point["caption"]+f" **labels:{labels_str}")
            wandb.log({"examples": images})
        if self.model_type == "lxmert":
            data_point = self.data_csv[idx]
            img_dir = os.path.join("/content/project/SSLMemes/data/Memotion2.0/images",f"{self.mode}_images")
            img_path = os.path.join(img_dir,data_point['image'])
            frcnn_cfg = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
            image_preprocess = Preprocess(frcnn_cfg)
            images, sizes, scales_yx = image_preprocess(img_path)
            _,c,w,h = images.size()
            images = np.array(images).reshape([w,h,c])
            img_index = self.img_name2index[data_point["image"]]
            bboxes = self.boxes[img_index]
            all_boxes = []
            for i, rect in enumerate(bboxes):
                x, y, w, h = rect.numpy()
                box_data = {"position" : {
                    "middle" : [x,y],
                    "width" : w,
                    "height" : h},
                    "domain" : "pixel"}
                all_boxes.append(box_data)
            
            box_image = wandb.Image(images, boxes = {"predictions": {"box_data": all_boxes}})
            table = wandb.Table(columns=["image"])
            table.add_data(img)

# load data
def collate_fn_batch_visualbert(batch,tokenizer=None):
    captions, img_features, labels = zip(*batch)
    toks = tokenizer.batch_encode_plus(
        list(captions), 
        max_length=32, 
        padding="max_length", 
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt")
    img_features = torch.stack(img_features, dim=0)
    labels = torch.tensor(labels)
    return toks, img_features, labels

def collate_fn_batch_visualbert_semi_supervised(batch,tokenizer=None):
    feature, labels = zip(*batch)
    supervised_image_feature = []
    supervised_text_feature = []
    unsupervised_image_feature = []
    unsupervised_text_feature = []
    for f in feature:
        supervised_image_feature.extend(f[0])
        supervised_text_feature.extend(f[1])
        unsupervised_image_feature.extend(f[2])
        unsupervised_text_feature.extend(f[3])

    supervised_toks = tokenizer.batch_encode_plus(
        list(supervised_text_feature), 
        max_length=32, 
        padding="max_length", 
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt")
    
    unsupervised_toks = tokenizer.batch_encode_plus(
        list(supervised_text_feature), 
        max_length=32, 
        padding="max_length", 
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt")

    supervised_img_features = torch.stack(supervised_image_feature, dim=0)
    unsupervised_img_features = torch.stack(unsupervised_image_feature, dim=0)
    labels = torch.tensor(labels)
    labels = labels.view(-1,labels.size()[-1])
    supervised_feature=[]
    unsupervised_feature=[]
    supervised_feature.append(supervised_img_features)
    supervised_feature.append(supervised_toks)
    unsupervised_feature.append(unsupervised_img_features)
    unsupervised_feature.append(unsupervised_toks)
    return supervised_feature,unsupervised_feature, labels

def collate_fn_batch_lxmert_semi_supervised(batch,tokenizer=None):
    feature, labels = zip(*batch)
    supervised_image_feature = []
    supervised_text_feature = []
    supervised_boxes_feature = []
    unsupervised_image_feature = []
    unsupervised_text_feature = []
    unsupervised_boxes_feature = []
    for f in feature:
        supervised_image_feature.extend(f[0])
        supervised_text_feature.extend(f[1])
        supervised_boxes_feature.extend(f[2])
        unsupervised_image_feature.extend(f[3])
        unsupervised_text_feature.extend(f[4])
        unsupervised_boxes_feature.extend(f[5])
    supervised_toks = tokenizer.batch_encode_plus(
        list(supervised_text_feature), 
        max_length=32, 
        padding="max_length", 
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt")
    
    unsupervised_toks = tokenizer.batch_encode_plus(
        list(supervised_text_feature), 
        max_length=32, 
        padding="max_length", 
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt")

    supervised_img_features = torch.stack(supervised_image_feature, dim=0)
    unsupervised_img_features = torch.stack(unsupervised_image_feature, dim=0)
    supervised_boxes_feature = torch.stack(supervised_boxes_feature)
    unsupervised_boxes_feature = torch.stack(unsupervised_boxes_feature)
    labels = torch.tensor(labels)
    labels = labels.view(-1,labels.size()[-1])
    supervised_feature=[]
    unsupervised_feature=[]
    supervised_feature.append(supervised_img_features)
    supervised_feature.append(supervised_toks)
    supervised_feature.append(supervised_boxes_feature)
    unsupervised_feature.append(unsupervised_img_features)
    unsupervised_feature.append(unsupervised_toks)
    unsupervised_feature.append(unsupervised_boxes_feature)
    return supervised_feature,unsupervised_feature, labels 

def collate_fn_batch_lxmert(batch,tokenizer=None):
    captions, boxes, img_features, labels = zip(*batch)
    toks = tokenizer.batch_encode_plus(
        list(captions), 
        max_length=32, 
        padding="max_length", 
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt")
    img_features = torch.stack(img_features, dim=0)
    boxes = torch.stack(boxes)
    labels = torch.tensor(labels)
    return toks, boxes, img_features, labels 

def collate_fn_batch_vilt(batch,processor=None):
    #"""
    imgs, captions, labels = zip(*batch)
    inputs = processor(images=list(imgs), text=list(captions), return_tensors="pt", 
            padding='max_length', truncation=True, add_special_tokens=True)
    #"""
    #print (inputs.input_ids.shape, inputs.pixel_values.shape)
    """
    inputs, labels = zip(*batch)
    inputs_ids = [i.input_ids for i in inputs]
    pixel_values = [i.pixel_values for i in inputs]
    for i in pixel_values:
        print (i.shape)
    """
    labels = torch.tensor(labels)
    return inputs.input_ids, inputs.pixel_values, labels
    #return torch.cat(inputs_ids, dim=0), torch.cat(pixel_values, dim=0), labels
if __name__ == '__main__':
    wandb.init(project="meme_experiments", entity="vietnguyen")
    data = ImageTextClassificationDataset("/content/project/SSLMemes/data/features/lxmert/", "/content/project/SSLMemes/data/splits/random/memotion_train.csv", 
                supervise = True,model_type="lxmert", vilt_processor=None,mode='train')
    data.show_image(1)
    data.show_image(2)
    data.show_image(3)
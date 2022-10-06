
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

class ImageTextClassificationDataset(Dataset):
    def __init__(self, img_feature_path, csv_path, model_type="visualbert", vilt_processor=None,mode='train'): 
        
        
        self.model_type = model_type

        if mode=='train':
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
       
        self.data_csv = []
        df = pd.read_csv(csv_path)
        for _,row in df.iterrows():
            data_item = {}
            data_item["image"] = str(row["Id"])+".jpg"
            data_item["caption"] = row["ocr_text"]
            labels = []
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
            #return inputs, data_point["label"]

    def __len__(self):
        return len(self.data_csv)



import torch 
from .dataClass import ImageTextClassificationForVLMDataset 
from functools import partial 

# load data
def collate_fn_batch_visualbert(batch, tokenizer=None, debug=False):
    if not debug:
        captions, img_features, labels = zip(*batch)
    elif debug:
        captions, img_features, labels, metadata, idx = zip(*batch)

    toks = tokenizer.batch_encode_plus(
        list(captions), 
        max_length=128, 
        padding="max_length", 
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt")
    img_features = torch.stack(img_features, dim=0)
    labels = torch.tensor(labels)

    if not debug:
        return toks, img_features, labels
    else:
        return toks, img_features, labels, metadata, idx

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

def create_loaders(cfg,model_type,processor,tokenizer):
    img_feature_path = cfg.img_feature_path
    dataset_train = ImageTextClassificationForVLMDataset(img_feature_path, cfg.train_csv_path, 
                supervise = not cfg.semi_supervised,model_type=model_type, vilt_processor=processor,mode='train')
    dataset_val = ImageTextClassificationForVLMDataset(img_feature_path, cfg.val_csv_path, model_type=model_type,mode='val')

    if cfg.semi_supervised:
        if model_type == "visualbert":
            collate_fn_batch = partial(collate_fn_batch_visualbert_semi_supervised,tokenizer=tokenizer)
        elif model_type == "lxmert":
            collate_fn_batch = partial(collate_fn_batch_lxmert_semi_supervised,tokenizer=tokenizer)
    else:
        if model_type == "visualbert":
            collate_fn_batch = partial(collate_fn_batch_visualbert,tokenizer=tokenizer)
        elif model_type == "lxmert":
            collate_fn_batch = partial(collate_fn_batch_lxmert,tokenizer=tokenizer)
        elif model_type == "vilt":
            collate_fn_batch = partial(collate_fn_batch_vilt,processor=processor)

    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        collate_fn = collate_fn_batch,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=3,)
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        collate_fn = collate_fn_batch,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=3,)
    return train_loader, val_loader

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
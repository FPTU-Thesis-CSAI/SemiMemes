import torch.optim as optim
from transformers import AutoConfig, BertTokenizer, VisualBertModel, \
        VisualBertForVisualReasoning, LxmertModel, LxmertTokenizer, LxmertConfig,VisualBertConfig
from model import ModelForBinaryClassification
from data import ImageTextClassificationDataset
from functools import partial 
from data import collate_fn_batch_visualbert,collate_fn_batch_lxmert,collate_fn_batch_vilt,collate_fn_batch_visualbert_semi_supervised,collate_fn_batch_lxmert_semi_supervised
import torch 

def build_optimizer(model, optimizer, learning_rate):
    if optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(),
                              lr=learning_rate, momentum=0.9)
    elif optimizer == "adam":
        optimizer = optim.Adam(model.parameters(),
                               lr=learning_rate)
    return optimizer

def create_model(cfg,model_type):
    # load model
    if model_type == "visualbert":
        config = VisualBertConfig.from_pretrained(cfg.model_path)
        cfg.update({"output_hidden":config.hidden_size})
        model = VisualBertModel.from_pretrained(cfg.model_path)
        model = ModelForBinaryClassification(cfg,model,num_label=4)
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        processor = None
    elif model_type == "lxmert":
        config = LxmertConfig.from_pretrained(cfg.model_path)
        cfg.update({"output_hidden":config.hidden_size})
        model = LxmertModel.from_pretrained(cfg.model_path)
        model = ModelForBinaryClassification(cfg,model,num_label=4)
        tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased") 
        processor = None
    elif model_type == "vilt":
        from transformers import ViltProcessor, ViltModel, ViltForImagesAndTextClassification
        config = AutoConfig.from_pretrained("dandelin/vilt-b32-mlm")
        config.num_images = 1
        model = ViltForImagesAndTextClassification(config)
        model.vilt = ViltModel.from_pretrained(cfg.model_path)
        processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
        tokenizer = None
    return model,tokenizer,processor

def create_loaders(cfg,model_type,processor,tokenizer):
    img_feature_path = cfg.img_feature_path
    dataset_train = ImageTextClassificationDataset(img_feature_path, cfg.train_csv_path, 
                supervise = not cfg.semi_supervised,model_type=model_type, vilt_processor=processor,mode='train')
    dataset_val = ImageTextClassificationDataset(img_feature_path, cfg.val_csv_path, model_type=model_type,mode='val')

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
    
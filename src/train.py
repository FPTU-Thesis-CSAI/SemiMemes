import os
import cv2
import json
import wandb
import argparse
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch
import torch.optim as optim
from torch.optim import Adam, Adadelta, Adamax, Adagrad, RMSprop, Rprop, SGD
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoConfig, BertTokenizer, VisualBertModel, \
        VisualBertForVisualReasoning, LxmertModel, LxmertTokenizer, LxmertConfig,VisualBertConfig
from data import ImageTextClassificationDataset
from eval import evaluate
from model import ModelForBinaryClassification
from data import collate_fn_batch_visualbert,collate_fn_batch_lxmert,collate_fn_batch_vilt,collate_fn_batch_visualbert_semi_supervised,collate_fn_batch_lxmert_semi_supervised
from functools import partial 

wandb.init()


def train(args, train_loader, val_loader, model, scaler=None, step_global=0, epoch=-1, \
        val_best_score=0, processor=None):
    model_type = args.model_type
    train_loss = 0
    train_steps = 0

    model.cuda()
    model.train()
    for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        optimizer.zero_grad()

        if model_type == "visualbert":
            batch_cap, batch_img, y = data
            batch_inputs = {}
            for k,v in batch_cap.items():
                batch_inputs[k] = v.cuda()
            img_attention_mask = torch.ones(batch_img.shape[:-1], dtype=torch.long)
            img_token_type_ids = torch.ones(batch_img.shape[:-1], dtype=torch.long)
            batch_inputs.update({
                "visual_embeds": batch_img.cuda(),
                "visual_token_type_ids": img_token_type_ids.cuda(),
                "visual_attention_mask": img_attention_mask.cuda(),
                })
        elif model_type == "lxmert":
            batch_cap, batch_box, batch_img, y = data
            batch_inputs = {}
            for k,v in batch_cap.items():
                batch_inputs[k] = v.cuda()
            batch_inputs.update({
                "visual_feats": batch_img.cuda(),
                "visual_pos": batch_box.cuda(),
                })
        elif model_type == "vilt":
            input_ids, pixel_values, y = data
        y = y.cuda()

        if args.amp:
            with autocast():
                if model_type in ["visualbert", "lxmert"]:
                    outputs = model(**batch_inputs, labels=y)
                elif model_type == "vilt":
                    outputs = model(input_ids=input_ids.cuda(), 
                        pixel_values=pixel_values.cuda(), labels=y)
        else:
            if model_type in ["visualbert", "lxmert"]:
                outputs = model(**batch_inputs, labels=y)
            elif model_type == "vilt":
                outputs = model(input_ids=input_ids.cuda(), 
                        pixel_values=pixel_values.cuda(), labels=y)
                #logits = outputs.logits
                #idx = logits.argmax(-1).item()
                #model.config.id2label[idx]

        loss = outputs.loss
        scores = outputs.logits
        wandb.log({"loss": loss})

        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        # log lr
        lr = optimizer.param_groups[0]['lr']
        wandb.log({"lr": lr})

        train_loss += loss.item()
        #wandb.log({"Loss": loss.item()})
        train_steps += 1
        step_global += 1

        # save model every K iterationsn
        if step_global % args.checkpoint_step == 0:
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint_iter_{step_global}")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            # if model_type == "visualbert":
            #     model.encoder.save_pretrained(checkpoint_dir)
            # elif model_type == "lxmert":
            #     model.encoder.save_pretrained(checkpoint_dir)
            # elif model_type == "vilt":
            #     processor.save_pretrained(checkpoint_dir)
            #     model.save_pretrained(checkpoint_dir)
            if model_type == "vilt":
                processor.save_pretrained(checkpoint_dir)
            checkpoint_path = os.path.join(checkpoint_dir,'saved_model')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, checkpoint_path)
        # evaluate and save
        if step_global % args.eval_step == 0:
            # evaluate
            print (f"====== evaluate ======")
            average_precison1, example_auc1, macro_auc1, micro_auc1,ranking_loss1,accuarcy, f_score_micro, f_score_macro, recall, precision,_ = evaluate(val_loader, model, model_type=model_type)
            # print(f"epoch:{epoch},global step:{step_global},val performance"
            #     +f"\naccuarcy:{accuarcy}\nf_score_micro:{f_score_micro}\nf_score_macro:{f_score_macro}"
            #     +f"\nrecall:{recall} \nprecision:{precision}"
            #     +f"\naverage_precison1:{average_precison1}\nexample_auc1:{example_auc1}"
            #     +f"\nmacro_auc1:{macro_auc1}\nmicro_auc1:{micro_auc1}\nranking_loss1:{ranking_loss1}")

            print(f"\nf_score_macro:{f_score_macro}")

            print (f"=======================")
            wandb.log({"eval_recall": recall})
            wandb.log({"eval_precision": precision})
            wandb.log({"eval_accuarcy": accuarcy})
            wandb.log({"eval_f_score_micro": f_score_micro})
            wandb.log({"eval_f_score_macro": f_score_macro})
            wandb.log({"eval_average_precison1": average_precison1})
            wandb.log({"eval_example_auc1": example_auc1})
            wandb.log({"eval_macro_auc1": macro_auc1})
            wandb.log({"eval_micro_auc1": micro_auc1})
            wandb.log({"eval_ranking_loss1": ranking_loss1})
            if val_best_score < f_score_macro:
                val_best_score = f_score_macro
            else:
                continue
            checkpoint_dir = os.path.join(args.output_dir, f"best_checkpoint")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            # if model_type == "visualbert":
            #     model.encoder.save_pretrained(checkpoint_dir)
            # elif model_type == "lxmert":
            #     model.encoder.save_pretrained(checkpoint_dir)
            # elif model_type == "vilt":
            #     processor.save_pretrained(checkpoint_dir)
            #     model.save_pretrained(checkpoint_dir)
            if model_type == "vilt":
                processor.save_pretrained(checkpoint_dir)
            checkpoint_path = os.path.join(checkpoint_dir,'saved_model')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'val_best_score':val_best_score
                }, checkpoint_path)
            print (f"===== best model saved! =======")
                
    train_loss /= (train_steps + 1e-9)
    return train_loss, step_global, val_best_score


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--img_feature_path', type=str,default="data/features/visualbert/")
    parser.add_argument('--train_csv_path', type=str, default="data/splits/random/memotion_train.csv")
    parser.add_argument('--val_csv_path', type=str, default="data/splits/random/memotion_val.csv")
    parser.add_argument('--model_type', type=str, default="visualbert", help="visualbert or lxmert or vilt")
    parser.add_argument('--model_path', type=str, default="uclanlp/visualbert-nlvr2-coco-pre")
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--eval_step', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--amp',type=bool,default=True, \
                help="automatic mixed precision training")
    parser.add_argument('--output_dir', type=str, default="./tmp")
    parser.add_argument('--checkpoint_step', type=int, default=1000)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--resume_training', type=bool, default=False)
    parser.add_argument('--semi-supervised', type=bool, default=False)
    # parser = argparse.ArgumentParser(description='train')
    # parser.add_argument('--img_feature_path', type=str, required=True)
    # parser.add_argument('--train_json_path', type=str, required=True)
    # parser.add_argument('--val_json_path', type=str, required=True)
    # parser.add_argument('--model_type', type=str, default="visualbert", help="visualbert or lxmert or vilt")
    # parser.add_argument('--model_path', type=str, default="uclanlp/visualbert-nlvr2-coco-pre")
    # parser.add_argument('--learning_rate', type=float, default=2e-5)
    # parser.add_argument('--epoch', type=int, default=100)
    # parser.add_argument('--eval_step', type=int, default=100)
    # parser.add_argument('--batch_size', type=int, default=64)
    # parser.add_argument('--amp', action="store_true", \
    #             help="automatic mixed precision training")
    # parser.add_argument('--output_dir', type=str, default="./tmp")
    # parser.add_argument('--checkpoint_step', type=int, default=100)
    # parser.add_argument('--random_seed', type=int, default=42)
    
    args = parser.parse_args()
    
    torch.manual_seed(args.random_seed)

    model_type = args.model_type
    # load model
    if model_type == "visualbert":
        config = VisualBertConfig.from_pretrained(args.model_path)
        model = VisualBertModel.from_pretrained(args.model_path)
        model = ModelForBinaryClassification(model,config)
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        processor = None
    elif model_type == "lxmert":
        config = LxmertConfig.from_pretrained(args.model_path)
        model = LxmertModel.from_pretrained(args.model_path)
        model = ModelForBinaryClassification(model,config)
        tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased") 
        processor = None
    elif model_type == "vilt":
        from transformers import ViltProcessor, ViltModel, ViltForImagesAndTextClassification
        config = AutoConfig.from_pretrained("dandelin/vilt-b32-mlm")
        config.num_images = 1
        model = ViltForImagesAndTextClassification(config)
        model.vilt = ViltModel.from_pretrained(args.model_path)
        processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
        tokenizer = None
    
    img_feature_path = args.img_feature_path
    dataset_train = ImageTextClassificationDataset(img_feature_path, args.train_csv_path, 
                supervise = not args.semi_supervised,model_type=model_type, vilt_processor=processor,mode='train')
    dataset_val = ImageTextClassificationDataset(img_feature_path, args.val_csv_path, model_type=model_type,mode='val')

    if args.semi_supervised:
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
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=3,)
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        collate_fn = collate_fn_batch,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=3,)
    
    # mixed precision training 
    if args.amp:
        scaler = GradScaler()
    else:
        scaler = None
    
    optimizer = optim.AdamW(
        [{'params': model.parameters()},], 
        lr=args.learning_rate)
    
    if args.resume_training:
        model.cuda()
        ck_path = os.path.join(args.output_dir, "best_checkpoint/saved_model")
        checkpoint = torch.load(ck_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        val_best_score = checkpoint['val_best_score']
        print(f"previous best checkpoint: epoch:{epoch}, loss:{loss}, val_best_score: {val_best_score}")

    global_step, val_best_score = 0, 0
    for epoch in range(args.epoch):
        loss, global_step, val_best_score = train(args, train_loader, val_loader, model, scaler=scaler, \
                step_global=global_step, epoch=epoch, val_best_score=val_best_score, processor=processor)
        print (f"epoch: {epoch}, global step: {global_step}, loss: {loss}")


import os
import wandb
import argparse
from copy import deepcopy
from tqdm.auto import tqdm
import torch
from torch.cuda.amp import autocast, GradScaler
from VLMeval import evaluate
from utils.VLMutils import build_optimizer
from data.dataLoaders import create_loaders
from model.VLM import create_model 
from functools import partial
import yaml 
from arguments import get_args 

def train_on_epoch(cfg, train_loader, val_loader, model,optimizer, scaler=None, step_global=0, epoch=-1, \
        val_best_score=0, processor=None):
    train_loss = 0
    train_steps = 0

    model.cuda()
    model.train()
    for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        optimizer.zero_grad()

        if cfg.model_type == "visualbert":
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
        elif cfg.model_type == "lxmert":
            batch_cap, batch_box, batch_img, y = data
            batch_inputs = {}
            for k,v in batch_cap.items():
                batch_inputs[k] = v.cuda()
            batch_inputs.update({
                "visual_feats": batch_img.cuda(),
                "visual_pos": batch_box.cuda(),
                })
        elif cfg.model_type == "vilt":
            input_ids, pixel_values, y = data
        y = y.cuda()

        if cfg.amp:
            with autocast():
                if cfg.model_type in ["visualbert", "lxmert"]:
                    outputs = model(**batch_inputs, labels=y)
                elif cfg.model_type == "vilt":
                    outputs = model(input_ids=input_ids.cuda(), 
                        pixel_values=pixel_values.cuda(), labels=y)
        else:
            if cfg.model_type in ["visualbert", "lxmert"]:
                outputs = model(**batch_inputs, labels=y)
            elif cfg.model_type == "vilt":
                outputs = model(input_ids=input_ids.cuda(), 
                        pixel_values=pixel_values.cuda(), labels=y)
                #logits = outputs.logits
                #idx = logits.argmax(-1).item()
                #model.config.id2label[idx]

        loss = outputs.loss
        scores = outputs.logits
        wandb.log({"loss": loss})

        if cfg.amp:
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
        if step_global % cfg.checkpoint_step == 0:
            checkpoint_dir = os.path.join(cfg.output_dir, f"checkpoint_iter_{step_global}")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            if cfg.model_type == "vilt":
                processor.save_pretrained(checkpoint_dir)
            checkpoint_path = os.path.join(checkpoint_dir,'saved_model')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, checkpoint_path)
        # evaluate and save
        if step_global % cfg.eval_step == 0:
            # evaluate
            print (f"====== evaliuate ======")
            average_precison1, example_auc1, macro_auc1, micro_auc1,ranking_loss1,accuarcy, f_score_micro, f_score_macro,recall, precision, roc_auc_score = evaluate(val_loader, model, model_type=cfg.model_type)
            print(f"epoch:{epoch},global step:{step_global},val performance"
                +f"\naccuarcy:{accuarcy}\nf_score_micro:{f_score_micro}\nf_score_macro:{f_score_macro}"
                +f"\nrecall:{recall} \nprecision:{precision}"
                +f"\naverage_precison1:{average_precison1}\nexample_auc1:{example_auc1}"
                +f"\nmacro_auc1:{macro_auc1}\nmicro_auc1:{micro_auc1}\nranking_loss1:{ranking_loss1}")
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
            checkpoint_dir = os.path.join(cfg.output_dir, f"best_checkpoint")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            if cfg.model_type == "vilt":
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

def train(config=None,args=None):
    # Initialize a new wandb run
    if args.use_sweep:
        with wandb.init(config=config):
            config = wandb.config
            config.update(args)
            model_type = config.model_type
            model,tokenizer,processor = create_model(config,model_type)
            train_loader, val_loader = create_loaders(config,model_type,processor,tokenizer)
            # mixed precision training 
            if config.amp:
                scaler = GradScaler()
            else:
                scaler = None
            
            optimizer = build_optimizer(model, config.optimizer, config.learning_rate)
            
            if config.resume_training:
                model.cuda()
                ck_path = os.path.join(config.output_dir, "best_checkpoint/saved_model")
                checkpoint = torch.load(ck_path)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                epoch = checkpoint['epoch']
                loss = checkpoint['loss']
                val_best_score = checkpoint['val_best_score']
                print(f"previous best checkpoint: epoch:{epoch}, loss:{loss}, val_best_score: {val_best_score}")

            global_step, val_best_score = 0, 0
            for epoch in range(config.epochs):
                loss, global_step, val_best_score = train_on_epoch(config, train_loader, val_loader, model,optimizer, scaler=scaler, \
                        step_global=global_step, epoch=epoch, val_best_score=val_best_score, processor=processor)
                print (f"epoch: {epoch}, global step: {global_step}, loss: {loss}")
    else:
        config.update(args)
        model_type = config.model_type
        model,tokenizer,processor = create_model(config,model_type)
        train_loader, val_loader = create_loaders(config,model_type,processor,tokenizer)
        # mixed precision training 
        if config.amp:
            scaler = GradScaler()
        else:
            scaler = None
        
        optimizer = build_optimizer(model, config.optimizer, config.learning_rate)
        
        if config.resume_training:
            model.cuda()
            ck_path = os.path.join(config.output_dir, "best_checkpoint/saved_model")
            checkpoint = torch.load(ck_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            loss = checkpoint['loss']
            val_best_score = checkpoint['val_best_score']
            print(f"previous best checkpoint: epoch:{epoch}, loss:{loss}, val_best_score: {val_best_score}")

        global_step, val_best_score = 0, 0
        for epoch in range(config.epochs):
            loss, global_step, val_best_score = train_on_epoch(config, train_loader, val_loader, model,optimizer, scaler=scaler, \
                    step_global=global_step, epoch=epoch, val_best_score=val_best_score, processor=processor)
            print (f"epoch: {epoch}, global step: {global_step}, loss: {loss}")

if __name__ == "__main__":
    args = get_args()
    torch.manual_seed(args.random_seed)
    
    if args.use_sweep:
        with open(args.hyper_yaml_path, 'r') as stream:
            hyper_config = yaml.safe_load(stream)
        sweep_id = wandb.sweep(hyper_config, project="meme_experiments")
        train = partial(train,args=args)
        wandb.agent(sweep_id, train, count=5)
    else:
        wandb.init(project="meme_experiments", entity="vietnguyen")
        with open(args.hyper_yaml_path, 'r') as stream:
            hyper_config = yaml.safe_load(stream)
        wandb.config.update(hyper_config)
        config = wandb.config
        train(config=config,args=args)


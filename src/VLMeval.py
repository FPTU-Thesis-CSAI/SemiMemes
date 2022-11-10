
import os
import argparse
from tqdm.auto import tqdm
import torch
from transformers import BertTokenizer, VisualBertModel, \
        VisualBertForVisualReasoning, LxmertForPreTraining, LxmertTokenizer,VisualBertConfig
from model.VLM import ModelForBinaryClassification
from data.dataClass import ImageTextClassificationForVLMDataset
from evaluation_metric import measure_average_precision
from evaluation_metric import measure_coverage
from evaluation_metric import measure_example_auc
from evaluation_metric import measure_macro_auc
from evaluation_metric import measure_micro_auc
from evaluation_metric import measure_ranking_loss
from evaluation_metric import f1_metric
from evaluation_metric import roc_auc

import numpy as np 

def evaluate(data_loader, model,threshold=0.5, model_type="visualbert", num_batch=None):
    model.cuda()
    model.eval()

    correct, total, all_true = 0, 0, 0
    preds = []
    total_preds = []
    total_y = []
    for i, data in tqdm(enumerate(data_loader), total=len(data_loader) if num_batch is None else num_batch):
    # for i, data in tqdm(enumerate(data_loader), total=1):

        if not num_batch is None and i >= num_batch:
            break

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

        elif  model_type == "lxmert":
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
        batch_inputs.update({"labels":y})
        with torch.no_grad():
            if model_type in ["visualbert", "lxmert"]:
                outputs = model(**batch_inputs)
            elif model_type == "vilt":
                batch_cap = input_ids.cuda()
                batch_img = pixel_values.cuda()
                outputs = model(input_ids=batch_cap, 
                        pixel_values=batch_img)
                #logits = outputs.logits
                #idx = logits.argmax(-1).item()
                #model.config.id2label[idx]

        scores = outputs.logits
        preds_current = torch.nn.Sigmoid()(scores) 
        pred_labels = preds_current >= threshold
        for yi,pi in zip(y,pred_labels):
            y_label = ''.join([str(i.detach().cpu().item()) for i in yi])
            p = ''.join([str(int(i)) for i in pi])
            if y_label == p:
                correct += 1 
        
        total_preds.append(preds_current)
        total_y.append(y)
        preds += pred_labels.cpu().numpy().tolist()
        total+=y.shape[0]
        all_true += sum(sum(y))
        

        # print errors
        #print (y != torch.argmax(scores, dim=1))
    total_preds = torch.cat(total_preds).detach().cpu().numpy()
    total_y = torch.cat(total_y).detach().cpu().numpy()

    # print(total_preds.shape)
    # print(total_y.shape)

    roc_auc_score = roc_auc.multilabel_binary_auroc(total_preds, total_y)

    # temp = total_preds[0]
    # for i in range(1, len(total_preds)):
    #     temp = np.vstack((temp, total_preds[i]))
    # total_preds = temp
    # temp = total_y[0]
    # for i in range(1, len(total_y)):
    #     temp = np.vstack((temp, total_y[i]))
    # total_y = temp
    accuarcy, f_score_micro, f_score_macro,recall, precision = f1_metric.metrics(total_preds,total_y)
    average_precison1 = measure_average_precision.average_precision(total_preds, total_y)
    example_auc1 = measure_example_auc.example_auc(total_preds, total_y)
    macro_auc1 = measure_macro_auc.macro_auc(total_preds, total_y)
    micro_auc1 = measure_micro_auc.micro_auc(total_preds, total_y)
    ranking_loss1 = measure_ranking_loss.ranking_loss(total_preds, total_y)
    return average_precison1, example_auc1, macro_auc1, micro_auc1,ranking_loss1,accuarcy, f_score_micro, f_score_macro,recall, precision, roc_auc_score
            

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='eval')
    # parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--img_feature_path', type=str,default="data/features/visualgenome/")
    parser.add_argument('--model_type', type=str, default='visualbert')
    parser.add_argument('--test_json_path', type=str, default="data/splits/random/memotion_val.csv")
    parser.add_argument('--output_preds', default="../tmp")
    parser.add_argument('--model_path', type=str, default="uclanlp/visualbert-vqa-coco-pre")    

    args = parser.parse_args()

    model_type = args.model_type
    # load model
    if model_type == "visualbert":
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        config = VisualBertConfig.from_pretrained(args.model_path)
        model = VisualBertModel.from_pretrained(args.model_path)
        model = ModelForBinaryClassification(model,config)
    elif model_type == "lxmert":
        model = LxmertForPreTraining.from_pretrained("unc-nlp/lxmert-base-uncased")
        model = ModelForBinaryClassification(model)
        tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased") 

    elif model_type == "vilt":
        from transformers import ViltProcessor, ViltForImagesAndTextClassification
        processor = ViltProcessor.from_pretrained(args.checkpoint_path)
        model = ViltForImagesAndTextClassification.from_pretrained(args.checkpoint_path)
    
    # load data
    def collate_fn_batch_visualbert(batch):
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
    
    def collate_fn_batch_lxmert(batch):
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

    def collate_fn_batch_vilt(batch):
        imgs, captions, labels = zip(*batch)
        inputs = processor(list(imgs), list(captions), return_tensors="pt", padding=True, truncation=True)
        labels = torch.tensor(labels)
        return inputs.input_ids, inputs.pixel_values.unsqueeze(1), labels
        

    img_feature_path = args.img_feature_path
    json_path = args.test_json_path
    if model_type in ["visualbert", "lxmert"]:
        dataset = ImageTextClassificationDataset(img_feature_path, json_path, model_type=model_type,mode='val')
    elif model_type == "vilt":
        dataset = ImageTextClassificationDataset(img_feature_path, json_path, model_type=model_type, vilt_processor=processor)
    if model_type == "visualbert":
        collate_fn_batch = collate_fn_batch_visualbert
    elif model_type == "lxmert":
        collate_fn_batch = collate_fn_batch_lxmert
    elif model_type == "vilt":
        collate_fn_batch = collate_fn_batch_vilt

    test_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn = collate_fn_batch,
        batch_size=16,
        shuffle=False,
        num_workers=16,)

    average_precison1, example_auc1, macro_auc1, micro_auc1,ranking_loss1,accuarcy, f_score_micro, f_score_macro,recall, precision,total_preds, roc_auc_score = evaluate(test_loader, model, model_type=model_type)
    # print (f"total example: {total}, # true example: {all_true}, acccuracy: {roc_auc_score}")
    print (f"acccuracy: {roc_auc_score}")

    # save preds
    # if args.output_preds:
    #     with open(os.path.join(args.checkpoint_path, "preds.txt"), "w") as f:
    #         for i in range(len(preds)):
    #             f.write(str(preds[i])+"\n")
        


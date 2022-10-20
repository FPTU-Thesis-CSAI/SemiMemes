from src.encoder import VISUAL_CONFIG
from src.encoder import VBFeatureExtraction as VBE

import os

import torch
import torch.nn as nn

def set_visual_config(args):
    VISUAL_CONFIG.l_layers = 9
    VISUAL_CONFIG.x_layers = 5
    VISUAL_CONFIG.r_layers = 5

class VBEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        # self.max_seq_length = args.max_seq_length
        set_visual_config(args)
        # self.tokenizer = BertTokenizer.from_pretrained(
        #         "bert-base-uncased",
        #         do_lower_case=True
        #     )
        self.model = VBE.from_pretrained(
            "bert-base-uncased")
        # if args.from_scratch:
        #     print("initializing all the weights")
        #     self.model.apply(self.model.init_bert_weights)

    @property
    def dim(self):
        return 768

    def forward(self, input_ids, token_type_ids, attention_mask, visual_embeds, visual_token_type_ids, visual_attention_mask):
        # train_features = convert_sents_to_features(
        #     sents, self.max_seq_length, self.tokenizer)

        # input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long).cuda()
        # input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long).cuda()
        # segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long).cuda()
        # assert feats.shape[1] == 36 , "Not Using 36 ROIs, please change the following 2 lines"
        # visual_segment_ids = torch.ones(input_ids.shape[0],feats.shape[1],dtype=torch.long).cuda()
        # v_mask = torch.ones(input_mask.shape[0],feats.shape[1],dtype=torch.long).cuda()

        output = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                            visual_feats=visual_embeds, visual_token_type_ids=visual_token_type_ids, visual_attention_mask=visual_attention_mask)
        return output

    def load(self, path):
        state_dict = torch.load(path)
        new_state_dict = {}
        print("Load VISUALBERT PreTrained Model from %s" % path)
        for key, value in state_dict.items():

            if key.startswith("bert."):
                new_state_dict[key[len("bert."):]] = value
            else:
                new_state_dict[key] = value

        state_dict = new_state_dict
        load_keys = set(state_dict.keys())
        model_keys = set(self.model.state_dict().keys())
        print()
        print("Weights in loaded but not in model:")
        for key in sorted(load_keys.difference(model_keys)):
            print(key)
        print()
        print("Weights in model but not in loaded:")
        for key in sorted(model_keys.difference(load_keys)):
            print(key)
        print()
        self.model.load_state_dict(state_dict, strict=False)


from torch import nn
from torch.nn import CrossEntropyLoss, KLDivLoss, LogSoftmax
from transformers.modeling_outputs import SequenceClassifierOutput
import torch 

class ModelForBinaryClassification(nn.Module):
    def __init__(self, encoder,config):
        super(ModelForBinaryClassification, self).__init__()
        self.encoder = encoder
        hidden_size = config.hidden_size
        self.ff = nn.Sequential(nn.Linear(hidden_size,hidden_size),nn.ReLU(),nn.Linear(hidden_size,4))

    def forward(self, input_ids, attention_mask, visual_feats, visual_pos, token_type_ids, labels=None):

        outputs = self.encoder(
                input_ids = input_ids,
                attention_mask = attention_mask,
                visual_feats = visual_feats,
                visual_pos = visual_pos,
                token_type_ids = token_type_ids)
        out_logits = self.ff(outputs["pooled_output"])

        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss(reduction='mean')
            loss = loss_fct(out_logits, labels.type(torch.DoubleTensor).to(out_logits.device))
        else:
            loss = None

        return SequenceClassifierOutput(
                loss=loss,
                logits=out_logits,
        )



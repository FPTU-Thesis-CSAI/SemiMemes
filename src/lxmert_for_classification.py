
from torch import nn
from torch.nn import CrossEntropyLoss, KLDivLoss, LogSoftmax
from transformers.modeling_outputs import SequenceClassifierOutput
import torch 

class LxmertForBinaryClassification(nn.Module):
    def __init__(self, lxmert):
        super(LxmertForBinaryClassification, self).__init__()
        self.lxmert = lxmert
        self.ff = nn.Sequential(nn.Linear(9500,3000),nn.ReLU(),nn.Linear(3000,4))

    def forward(self, input_ids, attention_mask, visual_feats, visual_pos, token_type_ids, labels=None):

        outputs = self.lxmert(
                input_ids = input_ids,
                attention_mask = attention_mask,
                visual_feats = visual_feats,
                visual_pos = visual_pos,
                token_type_ids = token_type_ids)
        reshaped_logits = outputs["question_answering_score"]
        out_logits = self.ff(reshaped_logits)

        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss(reduction='mean')
            loss = loss_fct(out_logits, labels.type(torch.DoubleTensor).to(out_logits.device))
        else:
            loss = None

        return SequenceClassifierOutput(
                loss=loss,
                logits=out_logits,
        )



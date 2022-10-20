
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

    def forward(self,**kwargs):
        labels = kwargs.pop('labels',None)   
        outputs = self.encoder(**kwargs)
        
        try:
            out_logits = self.ff(outputs["pooled_output"])
        except:
            out_logits = self.ff(outputs["pooler_output"])

        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss(reduction='mean')
            loss = loss_fct(out_logits, labels.type(torch.DoubleTensor).to(out_logits.device))
        else:
            loss = None

        return SequenceClassifierOutput(
                loss=loss,
                logits=out_logits,
        )


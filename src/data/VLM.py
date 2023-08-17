
from torch import nn
from torch.nn import CrossEntropyLoss, KLDivLoss, LogSoftmax
from transformers.modeling_outputs import SequenceClassifierOutput
import torch 
from transformers import AutoConfig, BertTokenizer, VisualBertModel, \
        VisualBertForVisualReasoning, LxmertModel, LxmertTokenizer, LxmertConfig,VisualBertConfig
        
class ModelForBinaryClassification(nn.Module):
    def __init__(self,cfg, encoder,num_label):
        super(ModelForBinaryClassification, self).__init__()
        self.encoder = encoder
        output_hidden = cfg.output_hidden
        self.num_label = num_label
        self.ff = nn.Sequential(nn.Linear(output_hidden,cfg.hidden_size),nn.Dropout(cfg.dropout),nn.Linear(cfg.hidden_size,self.num_label))

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
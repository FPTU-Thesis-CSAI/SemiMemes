from torch import nn
import torch
import torchvision.models as Models
import warnings
import torch.nn.functional as F 
from transformers import RobertaModel, DistilBertModel

warnings.filterwarnings("ignore", category=UserWarning) 

class TextfeatureNet(nn.Module):
    def __init__(self, args, neure_num,clip_dim=None,encoder_text=None):
        super(TextfeatureNet, self).__init__()
        self.args = args  
        self.encoder_text = encoder_text
        self.fc1 = nn.Linear(clip_dim,args.output_backbone_dim)
        self.fc2 = nn.Linear(clip_dim,args.output_backbone_dim)
        # self.dropout = nn.Dropout(0.2)
        # self.bigru = nn.LSTM(clip_dim,neure_num[-1], 1, bidirectional=False, batch_first=True, bias=False)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def forward(self, x=None, bert_emb=None,input_ids=None,attn_mask=None,clip_input_ids=None,clip_model=None,clip_emb=None):
        # with torch.no_grad():
        #     feats = clip_model.encode_text(clip_input_ids)
        text_feature = self.fc1(clip_emb)
        with torch.no_grad():
            text_encoded = self.encoder_text(clip_emb)
        text_encoded = self.fc1(text_encoded)
        text_encoded = self.dropout(self.relu(text_encoded))
        text_feature = self.dropout(self.relu(text_feature))
        # if self.args.use_norm:
        #     x = x / x.norm(dim=1, keepdim=True)
        #     x = self.dropout(x)
        # else:
        #     x = self.relu(x)
        #     x = self.dropout(x)
        return text_feature,text_encoded

class PredictNet(nn.Module):
    
    def __init__(self, args,neure_num, use_softmax=False):
        #print("---------PredictNet-----")
        super(PredictNet, self).__init__()
        self.mlp = make_predict_layers(neure_num)
        self.args = args
        # print("---------mlp----------",self.mlp)
        
        # if use_softmax:
        #     self.softmax = torch.nn.Softmax()
        #     self.sigmoid = None
        # else:
        #     self.softmax = None
        #     self.sigmoid = torch.nn.Sigmoid()


        #print("------------sigmoid-------------",self.sigmoid)
        # self.sigma = nn.Parameter(torch.FloatTensor([1.,1.,1./2]))

    def forward(self, x):
        y= self.mlp(x)
        return y


class AttentionNet(nn.Module):
    def __init__(self, neure_num):
        super(AttentionNet, self).__init__()
        self.mlp = make_layers(neure_num[:-1])
        self.attention = nn.Linear(neure_num[-2], neure_num[-1])
    def forward(self, x):
        temp_x = self.mlp(x)
        y = self.attention(temp_x)
        return y

class ImgNet(nn.Module):
    def __init__(self,args,clip_dim=None,encoder_img=None):
        super(ImgNet, self).__init__()
        self.encoder_img = encoder_img
        self.fc1 = nn.Sequential(       
            nn.Linear(clip_dim, args.output_backbone_dim)
            )
        self.fc2 = nn.Sequential(       
            nn.Linear(clip_dim, args.output_backbone_dim)
            )
        self.args = args 
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
            
    def forward(self, x):
        # with torch.no_grad():
        #     x = clip_model.encode_image(x) 
        #  
        image_feature = self.fc1(x)   
        with torch.no_grad():
            img_encoded = self.encoder_img(x)
        img_encoded = self.fc2(img_encoded)
        img_encoded = self.dropout(self.relu(img_encoded))
        image_feature = self.dropout(self.relu(image_feature))

        # if self.args.use_norm:
        #     x = x / x.norm(dim=1, keepdim=True)
        #     x = self.dropout(x)
        # else:          
        #     x = self.relu(x) 
        #     x = self.dropout(x)
        return image_feature,img_encoded

def make_layers(cfg):
    layers = []
    n = len(cfg)
    input_dim = cfg[0]
    for i in range(1, n):
        output_dim = cfg[i]
        layers += [nn.Linear(input_dim, output_dim)]
        input_dim = output_dim
    return nn.Sequential(*layers)

def make_predict_layers(cfg):
    layers = []
    n = len(cfg)
    input_dim = cfg[0]
    for i in range(1, n-1):
        output_dim = cfg[i]
        layers += [nn.Linear(input_dim, output_dim),nn.ReLU(True)]
        input_dim = output_dim
    layers.append(nn.Linear(cfg[-2], cfg[-1], bias=False))
    return nn.Sequential(*layers)

def Projector(mlp_expand_dim, embedding):
    mlp_spec = f"{embedding}-{mlp_expand_dim}"
    layers = []
    f = list(map(int, mlp_spec.split("-")))
    for i in range(len(f) - 2):
        layers.append(nn.Linear(f[i], f[i + 1]))
        layers.append(nn.BatchNorm1d(f[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(f[-2], f[-1], bias=False))
    return nn.Sequential(*layers)

class CmmlModel(nn.Module):
    def __init__(self,args,clip_model=None,cdim=None,image_encoder=None,text_encoder=None):
        super(CmmlModel, self).__init__()
        self.args = args 
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        param = args.Textfeaturepara.split(',')
        self.Textfeatureparam = list(map(int, param))
        param = args.Predictpara.split(',')
        self.Predictparam = list(map(int, param))
        param = args.Imgpredictpara.split(',')
        self.Imgpredictparam = list(map(int, param))
        param = args.Textpredictpara.split(',')
        self.Textpredictparam = list(map(int, param))
        param = args.Attentionparameter.split(',')
        self.Attentionparam = list(map(int, param))
        self.clip_model = clip_model
        self.cdim = cdim 
        self.generate_model()
    def generate_model(self):
        self.Textfeaturemodel = TextfeatureNet(self.args,self.Textfeatureparam,clip_dim=self.cdim,encoder_text=self.text_encoder)
        self.Imgpredictmodel = PredictNet(self.args,self.Imgpredictparam)
        self.Textpredictmodel = PredictNet(self.args,self.Textpredictparam)
        self.Predictmodel = PredictNet(self.args,self.Predictparam)
        self.Imgmodel = ImgNet(self.args,clip_dim=self.cdim,encoder_img = self.image_encoder)
        self.Attentionmodel = AttentionNet(self.Attentionparam)


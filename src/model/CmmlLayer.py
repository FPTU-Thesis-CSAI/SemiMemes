from torch import nn
import torch
import torchvision.models as Models
import warnings
import torch.nn.functional as F 
from transformers import RobertaModel, DistilBertModel
from .unsupervised import FusionNet

warnings.filterwarnings("ignore", category=UserWarning) 

class TextfeatureNet(nn.Module):
    
    def __init__(self, args, neure_num,clip_model=None,clip_dim=None):
        super(TextfeatureNet, self).__init__()
        self.args = args  
        if self.args.use_clip:
            self.clip_model = clip_model
            self.linear = nn.Linear(clip_dim,args.output_backbone_dim)
            # self.dropout = nn.Dropout(0.2)
            # self.bigru = nn.LSTM(clip_dim,neure_num[-1], 1, bidirectional=False, batch_first=True, bias=False)
        elif self.args.use_bert_model:
            if args.pretrain_bert_model == "distilbert-base-uncased":
                self.encoder = DistilBertModel.from_pretrained(args.pretrain_bert_model)
            if args.pretrain_bert_model == "roberta-base":
                self.encoder = RobertaModel.from_pretrained(args.pretrain_bert_model)
            self.linear = make_layers([768,neure_num[-1]])
            self.target_token_idx = 0
        else:
            self.mlp = make_layers(neure_num[:-1])
            self.feature = nn.Linear(neure_num[-2], neure_num[-1])
            if args.add_block_linear_bert_embed:
                self.linear = make_layers([384,neure_num[-2]])
        if self.args.use_drop_out:
            self.dropout = nn.Dropout(0.2)

    def forward(self, x=None, bert_emb=None,input_ids=None,attn_mask=None,clip_input_ids=None):
        if self.args.use_clip:
            with torch.no_grad():
                feats = self.clip_model.encode_text(clip_input_ids)
            if self.args.use_drop_out:
                feats = self.dropout(feats)
            x = self.linear(feats)
        elif self.args.use_bert_model:
            output = self.encoder(input_ids,attn_mask)
            last_hidden_state = output.last_hidden_state
            x = last_hidden_state[:, self.target_token_idx, :]
            x = self.linear(x)
            if self.args.use_drop_out:
                x = self.dropout(x)
        else:
            temp_x = self.mlp(x)
            if self.args.use_bert_embedding:
                if self.args.add_block_linear_bert_embed:
                    bert_emb = self.linear(bert_emb)
                temp_x = temp_x + bert_emb
            x = self.feature(temp_x)
        return x

class PredictNet(nn.Module):
    
    def __init__(self, args,neure_num, use_softmax=False):
        #print("---------PredictNet-----")
        super(PredictNet, self).__init__()
        self.mlp = make_predict_layers(neure_num)
        self.args = args
        if self.args.use_act:
            self.act = nn.ReLU()
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
        if self.args.use_act:
            x = self.act(x)
            y= self.mlp(x)
        else:
            y = self.mlp(x)
        # if not self.sigmoid is None:
        #     y = self.sigmoid(y)
        # else:
        #     y = self.softmax(y)
        #print("---------y------------", y)
        #print("--------------------")
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
    def __init__(self,args,clip_model=None,clip_dim=None):
        super(ImgNet, self).__init__()
        if args.use_clip:
            self.clip_model = clip_model
            self.fc1 = nn.Sequential(       
            nn.Linear(clip_dim, args.output_backbone_dim)
            )
        else:
            if args.resnet_model == 'resnet50':
                self.feature = Models.resnet50('ResNet50_Weights.DEFAULT')
                self.fc1 = nn.Sequential(       
                nn.Linear(2048, args.output_backbone_dim)
                )
            elif args.resnet_model == 'resnet18':
                self.feature = Models.resnet18('ResNet18_Weights.DEFAULT')
                self.fc1 = nn.Sequential(       
                nn.Linear(512, args.output_backbone_dim)
                )
            self.feature = nn.Sequential(*list(self.feature.children())[:-1])
        self.args = args 
        if args.use_drop_out:
            self.dropout = nn.Dropout(0.2)
            
    def forward(self, x):
        if self.args.use_clip:
            with torch.no_grad():
                x = self.clip_model.encode_image(x)
            if self.args.use_drop_out:
                x = self.dropout(x)
            x = self.fc1(x)
        else:
            N = x.size()[0]
            x = self.feature(x.view(N, 3, 256, 256))
            if self.args.resnet_model == 'resnet18':
                x = x.view(N, 512)
            elif self.args.resnet_model == 'resnet50':
                x = x.view(N, 2048)
            x = self.fc1(x)
            if self.args.use_drop_out:
                x = self.dropout(x)
        return x

def make_layers(cfg):
    layers = []
    n = len(cfg)
    input_dim = cfg[0]
    for i in range(1, n):
        output_dim = cfg[i]
        layers += [nn.Linear(input_dim, output_dim), nn.ReLU(inplace = True)]
        input_dim = output_dim
    return nn.Sequential(*layers)

def make_predict_layers(cfg):
    layers = []
    n = len(cfg)
    input_dim = cfg[0]
    for i in range(1, n):
        output_dim = cfg[i]
        layers += [nn.Linear(input_dim, output_dim)]
        input_dim = output_dim
    return nn.Sequential(*layers)

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

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

class VICReg(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_features = int(args.mlp_expand_dim.split("-")[-1])
        self.projector = Projector(args.mlp_expand_dim, args.mlp_expand_dim)

    def forward(self, x, y):
        x = self.projector(x)
        y = self.projector(y)

        if self.args.use_sim_loss:
            repr_loss = F.mse_loss(x, y)
        # x = torch.cat(FullGatherLayer.apply(x), dim=0)
        # y = torch.cat(FullGatherLayer.apply(y), dim=0)
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (self.args.batchsize - 1)
        cov_y = (y.T @ y) / (self.args.batchsize - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            self.num_features
        ) + off_diagonal(cov_y).pow_(2).sum().div(self.num_features)

        if self.args.use_sim_loss:
            loss = [
                self.args.std_coeff * std_loss,
                self.args.sim_coeff * repr_loss,
                self.args.cov_coeff * cov_loss
            ]
        else:
            loss = [
                self.args.std_coeff * std_loss,
                self.args.cov_coeff * cov_loss
            ]   
        return loss

class CmmlModel(nn.Module):
    def __init__(self,args,clip_model=None,cdim=None):
        super(CmmlModel, self).__init__()
        self.args = args 
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
        self.Textfeaturemodel = TextfeatureNet(self.args,self.Textfeatureparam,
            clip_model=self.clip_model,clip_dim=self.cdim)
        self.Imgpredictmodel = PredictNet(self.args,self.Imgpredictparam)
        self.Textpredictmodel = PredictNet(self.args,self.Textpredictparam)
        self.Predictmodel = PredictNet(self.args,self.Predictparam)
        self.Imgmodel = ImgNet(self.args,clip_model=self.clip_model,clip_dim=self.cdim)
        self.Attentionmodel = AttentionNet(self.Attentionparam)
        self.FusionCoattention = None
        self.ProjectormodelImgText = None 
        if self.args.use_coattention:
            self.FusionCoattention = FusionNet(self.Textfeatureparam[-1],self.Textfeatureparam[-1], 0.2)
        if self.args.use_one_head:
            self.ProjectormodelImgText = VICReg(self.args)


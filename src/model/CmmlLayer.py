from torch import nn
import torch
import torchvision.models as Models
import warnings
import torch.nn.functional as F 

warnings.filterwarnings("ignore", category=UserWarning) 

class TextfeatureNet(nn.Module):
    
    def __init__(self, args, neure_num):
        super(TextfeatureNet, self).__init__()
        self.mlp = make_layers(neure_num[:-1])
        self.feature = nn.Linear(neure_num[-2], neure_num[-1])
        self.args = args  

    def forward(self, x,bert_emb=None):
        temp_x = self.mlp(x)
        if self.args.use_bert_embedding:
            temp_x = temp_x + bert_emb
        x = self.feature(temp_x)
        return x

class PredictNet(nn.Module):
    
    def __init__(self, neure_num):
        #print("---------PredictNet-----")
        super(PredictNet, self).__init__()
        self.mlp = make_predict_layers(neure_num)
        # print("---------mlp----------",self.mlp)
        self.sigmoid = torch.nn.Sigmoid()
        #print("------------sigmoid-------------",self.sigmoid)
        self.sigma = nn.Parameter(torch.FloatTensor([1.,1.,1./2]))

    def forward(self, x):
        y = self.mlp(x)
        y = self.sigmoid(y)
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
    def __init__(self,args):
        super(ImgNet, self).__init__()
        self.feature = Models.resnet18('ResNet18_Weights.DEFAULT')
        self.feature = nn.Sequential(*list(self.feature.children())[:-1])
        self.fc1 = nn.Sequential(       
            nn.Linear(512, args.output_backbone_dim)
        )

    def forward(self, x):
        N = x.size()[0]
        x = self.feature(x.view(N, 3, 256, 256))
        x = x.view(N, 512)
        x = self.fc1(x)
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

def Projector(args, embedding):
    mlp_spec = f"{embedding}-{args.mlp_expand_dim}"
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
        self.projector = Projector(args, args.output_backbone_dim)

    def forward(self, x, y):
        x = self.projector(x)
        y = self.projector(y)

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

        loss = [
            self.args.std_coeff * std_loss,
            self.args.cov_coeff * cov_loss
        ]
        return loss

class CmmlModel(nn.Module):
    def __init__(self,args):
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
        self.generate_model()
    def generate_model(self):
        self.Textfeaturemodel = TextfeatureNet(self.args,self.Textfeatureparam)
        self.Imgpredictmodel = PredictNet(self.Imgpredictparam)
        self.Textpredictmodel = PredictNet(self.Textpredictparam)
        self.Predictmodel = PredictNet(self.Predictparam)
        self.Imgmodel = ImgNet(self.args)
        self.Attentionmodel = AttentionNet(self.Attentionparam)
        if self.args.use_vcreg_loss:
            self.Projectormodel = VICReg(self.args)
        else:
            self.Projectormodel = None


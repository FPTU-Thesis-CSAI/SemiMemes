import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

from exceptions.exceptions import InvalidBackboneError
from transformers import RobertaModel, DistilBertModel

from .utils import l2norm


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class Projectors(nn.Module):
    def __init__(self, i_dim, t_dim, out_dim):
        super(Projectors, self).__init__()
        self.t_dim = t_dim
        self.i_dim = i_dim
        self.out_dim = out_dim

        # Unsupervised Learning Head for Image
        self.img_projector = nn.Sequential(
                nn.Linear(i_dim, i_dim),
                nn.ReLU(inplace=True),
                nn.Linear(i_dim, out_dim)
        )

        # Unsupervised Learning Head for Text
        self.txt_projector = nn.Sequential(
                nn.Linear(in_features=t_dim, out_features=t_dim),
                nn.Tanh(),
                nn.Linear(in_features=t_dim, out_features=out_dim),
        )

    def forward(self, z_i, z_t):
        out_i = self.img_projector(z_i)
        out_t = self.txt_projector(z_t)
        return out_i, out_t

class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, txt_model, out_dim, dropout, pretrained=True):
        super(ResNetSimCLR, self).__init__()

        self.dropout = dropout
        self.out_dim = out_dim
        
        # ========== IMAGE ================
        if base_model == "resnet18":
            self.backbone = models.resnet18(pretrained=pretrained)
        elif base_model == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
        else:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")

        dim_mlp = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(Identity())

        # ========== TEXT =================
        self.text_encoder = RobertaModel.from_pretrained('roberta-base')  
        text_feature_dim = self.text_encoder.pooler.dense.out_features

        # ========== PROJECTOR ============
        self.projector = Projectors(dim_mlp, text_feature_dim, out_dim)

        self.maxpool = nn.MaxPool1d(2)

    def maxpool_two_views(self, z_a, z_b):
        z_a = z_a.unsqueeze(2)
        z_b = z_b.unsqueeze(2)
        z = self.maxpool(torch.cat((z_a, z_b), 2))
        z = torch.squeeze(z, 2)
        return z

    def encode_text(self, x_txt, x_att):
        _, text_feats = self.text_encoder(x_txt, x_att, return_dict=False)
        return text_feats

    def encode_image(self, x_img1):
        x_img1 = self.backbone(x_img1)
        return x_img1

    def forward(self, x_img1, x_img2, x_txt, x_att):
        txt_feats = self.encode_text(x_txt, x_att)
        img1_feats = self.encode_image(x_img1)
        img2_feats = self.encode_image(x_img2)
        mimg_feats = self.maxpool_two_views(img1_feats, img2_feats)

        mimg_feats, txt_feats = self.projector(mimg_feats, txt_feats)
        img2_feats = l2norm(img2_feats, -1)

        img1_feats = l2norm(img1_feats, -1)

        mimg_feats = l2norm(mimg_feats, -1)
        txt_feats = l2norm(txt_feats, -1)

        return txt_feats, img1_feats, img2_feats, mimg_feats


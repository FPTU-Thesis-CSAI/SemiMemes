import torch
from torch import nn
import torch.nn.functional as F
from .unsupervised import FusionNet 


class MultiModalClassifier(nn.Module):
    def __init__(self, fuse_type, model, num_classes, nl, bn=False):
        super(MultiModalClassifier, self).__init__()
        self.fuse_type = fuse_type
        self.text_embedding = model.text_embedding
        self.image_embedding = model.image_embedding
        self.im_gt_txt = self.image_embedding > self.text_embedding
        self.num_classes = num_classes
        self.nl_name = nl
        self.bn = bn
        if nl == 'relu':
            self.nl = nn.ReLU
        elif nl == 'tanh':
            self.nl = nn.Tanh
        elif nl == 'sigmoid':
            self.nl = nn.Sigmoid
        else:
            raise ValueError("Unsupported non-linearity {}".format(nl))
        self.embed_dim= min(self.text_embedding, self.image_embedding)
        self.projector = nn.Sequential(
                nn.Linear(
                    max(self.text_embedding, self.image_embedding),
                    self.embed_dim
                ),
                self.nl()
        )
        self.projector.apply(self.init_weights)

        input_dim = int(2 * self.embed_dim)
        if fuse_type == 'selfattend':
            self.soft = nn.Softmax(dim=1)
            self.single_embed = self.selfattend
            self.gen_key_L3 = nn.Linear(self.embed_dim, self.embed_dim // 2)
            self.gen_query_L3 = nn.Linear(self.embed_dim, self.embed_dim // 2)
        elif fuse_type == 'concat':
            self.single_embed = self.concat
        elif fuse_type == 'pie':
            self.single_embed = FusionNet(self.embed_dim, self.embed_dim, 0.35)
            input_dim = self.embed_dim
        elif fuse_type == 'mlb':
            self.mlb_module = nn.Sequential(
                nn.Linear(3 * self.embed_dim // 4, self.embed_dim // 2),
                nn.LayerNorm(self.embed_dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(self.embed_dim // 2, self.embed_dim // 2),
                nn.ReLU()
            )
            input_dim = self.embed_dim // 2
            self.mlb_tanh = nn.Tanh()
            self.single_embed = self.mlb
            self.feat1_proj = nn.Sequential(
                nn.Linear(self.embed_dim, 3 * self.embed_dim // 4),
                nn.ReLU(inplace=True)
            )
            self.feat2_proj = nn.Sequential(
                nn.Linear(self.embed_dim, 3 * self.embed_dim // 4),
                nn.ReLU(inplace=True)
            )

        else:
            raise ValueError(f"{fuse_type} is not supported.")

        self.classifier = nn.Sequential(
                    nn.Linear(input_dim, self.embed_dim),
                    nn.Identity() if not bn else nn.BatchNorm1d(self.embed_dim),
                    self.nl(),
                    nn.Linear(self.embed_dim, self.embed_dim // 2),
                    self.nl(),
                    nn.Linear(self.embed_dim // 2, num_classes)
        )
        self.classifier.apply(self.init_weights)
        self.name = self.get_name()

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.0)

    def get_name(self):
        name = 'classifier--2x{}{}--{}--{}--{}'.format(
                self.embed_dim, '--bn' if self.bn else '', self.fuse_type, self.nl_name, self.num_classes)
        return name

    def concat(self, vec1, vec2):
        return torch.cat((vec1, vec2), 1)

    def selfattend(self, vec1, vec2):
        q1 = F.relu(self.gen_query_L3(vec1))
        k1 = F.relu(self.gen_key_L3(vec1))
        q2 = F.relu(self.gen_query_L3(vec2))
        k2 = F.relu(self.gen_key_L3(vec2))
        score1 = torch.reshape(torch.bmm(q1.view(-1, 1, self.embed_dim // 2), k2.view(-1, self.embed_dim // 2, 1)), (-1, 1))
        score2 = torch.reshape(torch.bmm(q2.view(-1, 1, self.embed_dim // 2), k1.view(-1, self.embed_dim // 2, 1)), (-1, 1))
        wt_score1_score2_mat = torch.cat((score1, score2), 1)
        wt_i1_i2 = self.soft(wt_score1_score2_mat.float()) #prob
        prob_1 = wt_i1_i2[:,0]
        prob_2 = wt_i1_i2[:,1]
        wtd_i1 = vec1 * prob_1[:, None]
        wtd_i2 = vec2 * prob_2[:, None]
        out_rep = torch.cat((wtd_i1, wtd_i2), 1)
        return out_rep

    def mlb(self, vec1, vec2):
        out1 = self.feat1_proj(vec1)
        out2 = self.feat2_proj(vec2)
        z = torch.mul(out1, out2)
        z = self.mlb_tanh(z)
        z = self.mlb_module(z)
        return z

    
    def forward(self, z_img, z_txt):
        if self.im_gt_txt:
            z_img = self.projector(z_img)
        else:
            z_txt = self.projector(z_txt)
        z = self.single_embed(z_img, z_txt)
        return self.classifier(z)

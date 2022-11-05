import torch
from torch import nn
import torch.nn.functional as F
import timm
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer
import numpy as np


class MultiHeadSelfAttention(nn.Module):
  """Self-attention module by Lin, Zhouhan, et al. ICLR 2017"""
  def __init__(self, n_head, d_in, d_hidden):
    super(MultiHeadSelfAttention, self).__init__()

    self.n_head = n_head
    self.w_1 = nn.Linear(d_in, d_hidden, bias=False)
    self.w_2 = nn.Linear(d_hidden, n_head, bias=False)
    self.tanh = nn.Tanh()
    self.softmax = nn.Softmax(dim=1)
    self.init_weights()

  def init_weights(self):
    nn.init.xavier_uniform_(self.w_1.weight)
    nn.init.xavier_uniform_(self.w_2.weight)

  def forward(self, x, mask=None):
    if x.dim() == 2:
        x = x.unsqueeze(1)
    # This expects input x to be of size (b x seqlen x d_feat)
    attn = self.w_2(self.tanh(self.w_1(x)))
    if mask is not None:
        mask = mask.repeat(self.n_head, 1, 1).permute(1,2,0)
        attn.masked_fill_(mask, -np.inf)
    attn = self.softmax(attn)
    output = torch.bmm(attn.transpose(1,2), x)
    if output.shape[1] == 1:
        output = output.squeeze(1)
    return output, attn



class CLIPProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim,
        dropout,
        nl = None                   # Not Used
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim,
        dropout = 0.1,              # Not Used
        nl = 'relu'
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        if nl == 'relu':
            self.nl = nn.ReLU()
        elif nl == 'gelu':
            self.nl = nn.GELU()
        elif nl == 'tanh':
            self.nl = nn.Tanh()
        else:
            self.nl = nl
        self.fc = nn.Linear(projection_dim, projection_dim)

    def forward(self, x):
        x = self.projection(x)
        x = self.nl(x)
        x = self.fc(x)
        return x


class PIENet(nn.Module):
    """Polysemous Instance Embedding (PIE) module in Song and Soleymani 2019"""
    def __init__(self, d_in, d_out, dropout=0.0, n_embeds=1):
        super(PIENet, self).__init__()
        d_h = int((d_in + d_out) / 2)
        self.num_embeds = n_embeds
        self.attention = MultiHeadSelfAttention(n_embeds, d_in, d_h)
        self.fc_attn = nn.Linear(d_in, d_in)
        self.fc_proj = nn.Linear(d_in, d_out)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_in)
        self.init_weights()

    def init_weights(self):
        for fc in [self.fc_attn, self.fc_proj]:
            nn.init.xavier_uniform_(fc.weight)
            nn.init.constant_(fc.bias, 0.0)

    def forward(self, x, pad_mask=None):
        residual, _ = self.attention(x, pad_mask)
        residual = self.dropout(self.sigmoid(self.fc_attn(residual)))
        out = self.layer_norm(x + residual)
        out = self.fc_proj(out)
        return out

class FusionNet(nn.Module):
    """ Co-attention for MultiModality """
    def __init__(self, d_in, d_out, dropout):
        super().__init__()
        self.attention = MultiHeadSelfAttention(2, d_in, d_in)
        self.fc = nn.Linear(d_in, d_out)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_out)
        self.init_weights()
  
    def init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, img, txt):
        x = torch.cat((img.unsqueeze(1), txt.unsqueeze(1)), dim=1)
        residual, _  = self.attention(x)
        out = self.layer_norm(x + residual).sum(1)
        return out


class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """
    def __init__(
        self,
        model_name,
        pretrained=True,
        trainable=True,
        use_pie=True
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = trainable
        self.use_pie = use_pie
        if use_pie:
            self.image_head = PIENet(512, 512)

    def forward(self, x):
        out = self.model(x)
        if self.use_pie:
            out = self.image_head(out)
        return out


class TextEncoder(nn.Module):
    def __init__(
            self,
            model_name,
            pretrained=True,
            trainable=True,
            use_pie=True
    ):
        super().__init__()
        if pretrained:
            self.model = DistilBertModel.from_pretrained(model_name)
        else:
            self.model = DistilBertModel(config=DistilBertConfig())
            
        for p in self.model.parameters():
            p.requires_grad = trainable

        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0
        self.use_pie = use_pie
        if use_pie:
            self.text_head = PIENet(768,768)

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        out = last_hidden_state[:, self.target_token_idx, :]
        if self.use_pie:
            out = self.text_head(out)
        return out



class UnsupervisedModel(nn.Module):
    def __init__(
        self,
        image_encoder,
        text_encoder,
        projection_dim=512,
        projection_dropout=0.1,
        projection_type='std',
        pretrained=True,
        trainable=True
    ):
        super().__init__()
        if projection_type in ['std', 'pie']:
            projection_head = ProjectionHead
        elif projection_type == 'clip':
            projection_head = CLIPProjectionHead
        else:
            raise ValueError(f"{projection_type} not defined.")
        use_pie = projection_type == 'pie'
        if use_pie:
            self.fuse_net = FusionNet(projection_dim, projection_dim, projection_dropout)
        else:
            self.fuse_net = None

        self.image_encoder = ImageEncoder(image_encoder, pretrained, trainable, use_pie)
        self.text_encoder = TextEncoder(text_encoder, pretrained, trainable, use_pie)
        self.image_embedding = self.image_encoder.model.num_features
        self.text_embedding = 768
        self.projection_type = projection_type

        self.image_projection = projection_head(self.image_embedding, projection_dim, projection_dropout)
        self.text_projection = projection_head(self.text_embedding, projection_dim, projection_dropout)
        self.maxpool = nn.MaxPool1d(2)
        self.name = self.get_name(image_encoder, text_encoder, projection_dim, projection_dropout, projection_type)
        print(self)

    def get_name(self, image_encoder, text_encoder, projection_dim, projection_dropout, projection_type):
        return '{}--{}--2x{}d--{:.2f}p--{}'.format(
                image_encoder, text_encoder, projection_dim, projection_dropout, projection_type)

    def maxpool_two_views(self, z_a, z_b):
        z_a = z_a.unsqueeze(2)
        z_b = z_b.unsqueeze(2)
        z = self.maxpool(torch.cat((z_a, z_b), 2))
        z = torch.squeeze(z, 2)
        return z

    def forward(self, batch):
        image_features = self.image_encoder(batch['image1'])
        text_features = self.text_encoder(batch['text'], batch['mask'])

        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        if self.fuse_net:
            fusion1 = self.fuse_net(image_embeddings, text_embeddings)
        else:
            fusion1 = None

        if batch['image2'] is None:
            image2_embeddings = None
            mix_embeddings = image_embeddings
            fusion2 = None
            fusion = fusion1
        else:
            image2_embeddings = self.image_projection(self.image_encoder(batch['image2']))
            mix_embeddings = self.maxpool_two_views(image_embeddings, image2_embeddings)
            if self.fuse_net:
                fusion2 = self.fuse_net(image2_embeddings, text_embeddings)
                fusion = self.fuse_net(mix_embeddings, text_embeddings)
                fusion2 = F.normalize(fusion2)
                fusion = F.normalize(fusion)
            else:
                fusion2 = None
                fusion = None
            image2_embeddings = F.normalize(image2_embeddings)

        mix_embeddings = F.normalize(mix_embeddings)
        image_embeddings = F.normalize(image_embeddings)
        text_embeddings = F.normalize(text_embeddings)
        if self.fuse_net:
            fusion1 = F.normalize(fusion1)

        out = {
                'image1': image_embeddings,
                'image2': image2_embeddings,
                'image': mix_embeddings,
                'text': text_embeddings,
                'fusion1': fusion1,
                'fusion2': fusion2,
                'fusion': fusion,
                'label': batch['label']
        }
        return out


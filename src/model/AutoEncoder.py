import torch 
import torch.nn as nn
import torch.nn.functional as F  

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def Projector(mlp_expand_dim, embedding):
    mlp_spec = f"{embedding}-{mlp_expand_dim}-{mlp_expand_dim}-{mlp_expand_dim}"
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
        self.projector = Projector(args.mlp_expand_dim, args.output_backbone_dim)

    def forward(self, x):
        x = self.projector(x)

        # x = torch.cat(FullGatherLayer.apply(x), dim=0)
        # y = torch.cat(FullGatherLayer.apply(y), dim=0)
        x = x - x.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 

        cov_x = (x.T @ x) / (self.args.batchsize - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(self.num_features) 

        loss = [
            self.args.std_coeff * std_loss,
            self.args.cov_coeff * cov_loss
            ]   
        return loss

class AutoEncoder(nn.Module):
    def __init__(self,args, encode_image=False, encode_text=False):
        super().__init__()
        assert bool(encode_image) != bool(encode_text), "at least one modality"

        self.project_dim = 768

        self.encoder = torch.nn.Sequential(
            # torch.nn.Linear(768, 768),
            # torch.nn.ReLU(),
            torch.nn.Linear(768, self.project_dim),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(self.project_dim, 768),
            # torch.nn.ReLU(),
            # torch.nn.Linear(768, 768),
        )
        # self.vcreg = VICReg(args)
        self.encode_image=encode_image
        self.encode_text=encode_text

    def forward(self, x):
        encoded = self.encoder(x)
        # encoded = encoded / encoded.norm(dim=1, keepdim=True)
        decoded = self.decoder(encoded)
        # vcreg = self.vcreg(decoded)
        return decoded

class ModelCombineAE(nn.Module):
    def __init__(self, image_encoder, text_encoder):
        super().__init__()
        self.selu = nn.SELU()
        self.project_dim = 256
        self.fc_img = nn.Linear(768, self.project_dim)
        self.fc_text = nn.Linear(768, self.project_dim)
        self.fc_img_encoded = nn.Linear(768, self.project_dim)
        self.fc_text_encoded = nn.Linear(768, self.project_dim)
        
        self.img_dropout = nn.Dropout(0.2)
        self.text_dropout = nn.Dropout(0.2)
        self.img_encoded_dropout = nn.Dropout(0.2)
        self.text_encoded_dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(self.project_dim*4, 4)
        # self.fc_taskB = nn.Linear(self.project_dim*4, 4)

        self.encoder_img = image_encoder
        self.encoder_text = text_encoder


    def forward(self, img, text):
        image_feature = self.fc_img(img)
        text_feature = self.fc_text(text)
        
        image_feature = self.img_encoded_dropout(image_feature / image_feature.norm(dim=1, keepdim=True))
        text_feature = self.text_encoded_dropout(text_feature / text_feature.norm(dim=1, keepdim=True))

        with torch.no_grad():
            img_encoded = self.encoder_img(img)
            text_encoded = self.encoder_text(text)

        img_encoded = self.fc_img_encoded(img_encoded)
        text_encoded = self.fc_text_encoded(text_encoded)

        img_encoded = self.img_encoded_dropout(img_encoded / img_encoded.norm(dim=1, keepdim=True))
        text_encoded = self.text_encoded_dropout(text_encoded / text_encoded.norm(dim=1, keepdim=True))

        # img_encoded = self.img_encoded_dropout(self.selu(img_encoded))
        # text_encoded = self.text_encoded_dropout(self.selu(text_encoded))

        # image_feature = self.img_dropout(self.selu(image_feature))
        # text_feature = self.text_dropout(self.selu(text_feature))

        batch_feature = torch.cat((image_feature, text_feature, img_encoded, text_encoded), dim=1)

        y = self.fc(batch_feature)
        # y = self.sigmoid(y)

        return y

class ModelConCat(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.project_dim = 256
        self.fc_img = nn.Linear(768, self.project_dim)
        self.fc_text = nn.Linear(768, self.project_dim)
        self.fc_img_encoded = nn.Linear(768, self.project_dim)
        self.fc_text_encoded = nn.Linear(768, self.project_dim)
        
        self.img_dropout = nn.Dropout(0.2)
        self.text_dropout = nn.Dropout(0.2)
        self.img_encoded_dropout = nn.Dropout(0.2)
        self.text_encoded_dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(self.project_dim*2, 1)

        # self.encoder_img = image_encoder
        # self.encoder_text = text_encoder


    def forward(self, img, text):
        image_feature = self.fc_img(img)
        text_feature = self.fc_text(text)

        # with torch.no_grad():
        #     img_encoded = self.encoder_img(img)
        #     text_encoded = self.encoder_text(text)

        # img_encoded = self.fc_img_encoded(img_encoded)
        # text_encoded = self.fc_text_encoded(text_encoded)

        # img_encoded = self.img_encoded_dropout(self.relu(img_encoded))
        # text_encoded = self.text_encoded_dropout(self.relu(text_encoded))

        image_feature = self.img_dropout(self.relu(image_feature))
        text_feature = self.text_dropout(self.relu(text_feature))

        batch_feature = torch.cat((image_feature, text_feature), dim=1)

        y = self.fc(batch_feature)
        # y = self.sigmoid(y)

        return y
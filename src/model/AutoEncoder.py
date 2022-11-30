import torch 
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, encode_image=False, encode_text=False):
        super().__init__()
        assert bool(encode_image) != bool(encode_text), "at least one modality"

        self.project_dim = 768

        self.encoder = torch.nn.Sequential(
            # torch.nn.Linear(768, 768),
            # torch.nn.ReLU(),
            torch.nn.Linear(768, self.project_dim),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.PReLU(),
            torch.nn.Linear(self.project_dim, 768),
            # torch.nn.ReLU(),
            # torch.nn.Linear(768, 768),
        )

        self.encode_image=encode_image
        self.encode_text=encode_text

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
class ModelCombineAE(nn.Module):
    def __init__(self, image_encoder, text_encoder):
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
        self.fc = nn.Linear(self.project_dim*4, 4)

        self.encoder_img = image_encoder
        self.encoder_text = text_encoder


    def forward(self, img, text):
        image_feature = self.fc_img(img)
        text_feature = self.fc_text(text)

        with torch.no_grad():
            img_encoded = self.encoder_img(img)
            text_encoded = self.encoder_text(text)

        img_encoded = self.fc_img_encoded(img_encoded)
        text_encoded = self.fc_text_encoded(text_encoded)

        img_encoded = self.img_encoded_dropout(self.relu(img_encoded))
        text_encoded = self.text_encoded_dropout(self.relu(text_encoded))

        image_feature = self.img_dropout(self.relu(image_feature))
        text_feature = self.text_dropout(self.relu(text_feature))

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
        self.fc = nn.Linear(self.project_dim*2, 4)

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
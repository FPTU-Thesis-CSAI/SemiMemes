from torch import optim
from torch import nn
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from test_func import test_auto_encoder
from utils.unsupervisedUtils import EarlyStopping

# Train model

def train_auto_encoder(model, train_loader, val_loader, cuda=False, verbose=1):

    optimizer = optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-4)
    loss_func = nn.MSELoss()
    scheduler = StepLR(optimizer, step_size = 1, gamma = 0.95)
    list_train_loss, list_val_loss = [], []
    
    earlystopping = EarlyStopping(patience=10, verbose=True, delta=1e-4)
    
    print('============== Pretrain Auto Encoder ==================')

    for epoch in range(1000):
        model.train()
        epoch_loss = 0  
        for ii, (image_feature, text_feature) in tqdm(enumerate(train_loader), total = len(train_loader)):
            image_feature = image_feature.float()
            text_feature = text_feature.float()
            
            if cuda:
                image_feature = image_feature.cuda()
                text_feature = text_feature.cuda()

            if model.encode_image:
                text_feature_pred = model(image_feature)                
                loss = loss_func(text_feature_pred, text_feature)
            elif model.encode_text:
                image_feature_pred = model(text_feature)                
                loss = loss_func(image_feature_pred, image_feature)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        epoch_loss = epoch_loss/len(train_loader)

        val_loss = test_auto_encoder(model, val_loader)
        
        if epoch % verbose == 0:
            print('Epoch: {} - Train Loss: {:.6f}'.format(epoch + 1, epoch_loss))
            print('Epoch: {} - Valid Loss: {:.6f}'.format(epoch + 1, val_loss))

        list_train_loss.append(epoch_loss)
        list_val_loss.append(val_loss)
        
        earlystopping(val_loss)
        if earlystopping.early_stop:
            break

        # if val_loss > best_loss:
        #     trigger_times += 1
        #     print('Trigger Times:', trigger_times)
        # else:
        #     best_loss = val_loss
        #     trigger_times = 0

        # if trigger_times >= patience:
        #     print('Early stopping!')
        #     break

    return list_train_loss, list_val_loss
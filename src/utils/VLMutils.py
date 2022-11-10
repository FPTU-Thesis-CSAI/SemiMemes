import torch.optim as optim

def build_optimizer(model, optimizer, learning_rate):
    if optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(),
                              lr=learning_rate, momentum=0.9)
    elif optimizer == "adam":
        optimizer = optim.Adam(model.parameters(),
                               lr=learning_rate)
    return optimizer




    
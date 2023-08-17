import torch.nn as nn 
from copy import deepcopy
import torch 

class EMAN(nn.Module):
    def __init__(self,model=None,momentum=0.999,device = None):
        super(EMAN, self).__init__()
        self.momentum = momentum
        self.module = deepcopy(model)
        self.module.eval()
        self.device = device 
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model,mode=None):
        if mode == 'eman':
            with torch.no_grad():
                for (k_eman, v_eman),(k_main, v_main) in zip(self.module.state_dict().items(),model.state_dict().items()):
                    assert k_main == k_eman, "state_dict names are different!"
                    assert v_main.shape == v_eman.shape, "state_dict shapes are different!"
                    if 'num_batches_tracked' in k_eman:
                        v_eman.copy_(v_main)
                    else:
                        v_eman.copy_(v_eman*self.momentum+(1.-self.momentum)*v_main) 
                        
        elif mode == 'set':
            with torch.no_grad():
                for eman_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                    if self.device is not None:
                        model_v = model_v.to(device=self.device)
                    eman_v.copy_(model_v)   

    def update(self, model):
        self._update(model, mode='eman')

    def set(self, model):
        self._update(model, mode='set')
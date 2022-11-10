from torchvision.models import resnet50
import torch 
import torch.nn as nn  
from torchvision.models.feature_extraction import create_feature_extractor

class MultiScaleFE(torch.nn.Module):
    def __init__(self, base, layers, strides, size, last_ouput_depth=768, pretrain='IMAGENET1K_V2'):
        super(MultiScaleFE, self).__init__()
        # Get a resnet50 backbone
        assert base == 'resnet50', "only support resnet50"
        self.m = resnet50(weights=pretrain)
        self.size = size
        assert size == 512 or size == 384, "only support input size 512 or 384"

        self.layers = {k:k for k in layers}
        self.strides = strides
        assert len(self.layers.keys()) == 3 and len(self.strides) == 3, "only support 3 scales"
        
        self.body = create_feature_extractor(m, return_nodes=self.layers)
        self.conv_1x1 = [nn.Conv2d(last_input_depth, last_ouput_depth, 1, stride=last_stride) 
                            for (last_input_depth, last_stride) in zip ([512, 1024, 2048], self.strides)]
        # self.relu = nn.ReLU()

    def forward(self, x):
        assert (x.shape[-1] == self.size and x.shape[-2] == self.size), "do not match defined input size"
        self.feature_map = self.body(x)
        output = [self.conv_1x1[i](self.feature_map[layer]) for i, layer in enumerate(self.layers)]
        # output = [self.relu(o) for o in output]

        output = [torch.flatten(o, start_dim=-2) for o in output]
        output = torch.cat(output, dim=-1)

        # compatible with attention
        output = output.permute((0, 2, 1))

        return output

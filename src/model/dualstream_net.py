import torch
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models import resnet50, resnet34
from torch import nn
from transformers import LxmertModel

class MultiScaleFE(torch.nn.Module):
    def __init__(self, base, layers, strides, size, last_ouput_depth=768, pretrain='IMAGENET1K_V2'):
        super(MultiScaleFE, self).__init__()
        # Get a resnet50 backbone
        # assert base == 'resnet50', "only support resnet50"
        backbone = resnet34(pretrained = True)
        self.size = size
        assert size == 512 or size == 384, "only support input size 512 or 384"

        self.layers = {k:k for k in layers}
        self.strides = strides
        # assert len(self.layers.keys()) == 3 and len(self.strides) == 3, "only support 3 scales"
        
        self.body = create_feature_extractor(backbone, return_nodes=self.layers)
        # self.conv_1x1 = nn.ModuleList([nn.Conv2d(last_input_depth, last_ouput_depth, 1, stride=last_stride) 
        #                     for (last_input_depth, last_stride) in zip ([512, 1024, 2048], self.strides)])
        self.conv_1x1 = nn.ModuleList([nn.Conv2d(last_input_depth, last_ouput_depth, 1, stride=last_stride) 
                            for (last_input_depth, last_stride) in zip ([128, 256], self.strides)])

        self.relu = nn.ReLU()

    def forward(self, x):
        assert (x.shape[-1] == self.size and x.shape[-2] == self.size), "do not match defined input size"
        self.feature_map = self.body(x)
        output = [self.conv_1x1[i](self.feature_map[layer]) for i, layer in enumerate(self.layers)]
        output = [self.relu(o) for o in output]

        output = [torch.flatten(o, start_dim=-2) for o in output]
        output = torch.cat(output, dim=-1)

        # compatible with attention
        output = output.permute((0, 2, 1))

        return output

class PredictNet(nn.Module):
    
    def __init__(self, neure_num, use_softmax=False):
        #print("---------PredictNet-----")
        super(PredictNet, self).__init__()
        self.mlp = make_predict_layers(neure_num)
        # print("---------mlp----------",self.mlp)
        
        if use_softmax:
            self.softmax = torch.nn.Softmax()
            self.sigmoid = None
        else:
            self.softmax = None
            self.sigmoid = torch.nn.Sigmoid()


        #print("------------sigmoid-------------",self.sigmoid)
        self.sigma = nn.Parameter(torch.FloatTensor([1.,1.,1./2]))

    def forward(self, x):
        y = self.mlp(x)
        if not self.sigmoid is None:
            y = self.sigmoid(y)
        else:
            y = self.softmax(y)
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
    
    
class DualStream(torch.nn.Module):
    def __init__(self, args):
        super(DualStream, self).__init__()
        # self.img_encoder = MultiScaleFE(base='resnet50', size = 384, layers=['layer2', 'layer3', 'layer4'], strides=[4, 2, 1], last_ouput_depth=2048)
        self.img_encoder = MultiScaleFE(base='resnet34', size = 384, layers=['layer2', 'layer3'], strides=[4, 4], last_ouput_depth=2048)

        self.encoder = LxmertModel.from_pretrained("unc-nlp/lxmert-base-uncased")
        self.visual_pool = nn.AdaptiveAvgPool1d(1)
        
        self.args = args
        
    def forward(self, visual_img, input_ids, attention_mask=None, visual_attention_mask=None):
        visual_feats = self.img_encoder(visual_img)
        visual_pos = torch.zeros(visual_feats.shape[0], visual_feats.shape[1], 4)
        if visual_feats.is_cuda:
            visual_pos = visual_pos.cuda()
            
        output = self.encoder(input_ids=input_ids, visual_feats=visual_feats, visual_pos=visual_pos, attention_mask=attention_mask, visual_attention_mask=visual_attention_mask)
        
        language_output = output.pooled_output
        
        visual_output = output.vision_output.permute((0,2,1))
        visual_output = self.visual_pool(visual_output)
        visual_output = visual_output.squeeze(-1)
        
        return visual_output, language_output
    
class CmmlModel_v2(nn.Module):
    def __init__(self,args):
        super(CmmlModel_v2, self).__init__()
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
        # self.Imgmodel = ImgNet(self.args)
        # self.Textfeaturemodel = TextfeatureNet(self.args,self.Textfeatureparam)
        
        self.ImgTextModel = DualStream(self.args)
        self.Imgpredictmodel = PredictNet(self.Imgpredictparam)
        self.Textpredictmodel = PredictNet(self.Textpredictparam)
        self.Predictmodel = PredictNet(self.Predictparam)
        
        self.Attentionmodel = AttentionNet(self.Attentionparam)
        if self.args.use_vcreg_loss:
            self.Projectormodel = VICReg(self.args)
        else:
            self.Projectormodel = None

    
def main():
    net = DualStream()
    input_ids = torch.tensor([[1, 2, 3, 4], [4, 5, 6, 7]])
    img = torch.rand(2, 3, 384, 384)
    # output = net(input_ids, img)
    visual_output, language_output = net(input_ids, img)
    return 0
    
if __name__ == '__main__':
    main()
        
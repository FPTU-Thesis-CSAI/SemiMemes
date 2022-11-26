from torch.autograd import Variable
import torch 
import torch.nn as nn  
import torch.nn.functional as F 
import mmcv 
import numpy as np  

def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()

def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Avarage factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss

def _expand_binary_labels(labels, label_weights, label_channels):
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    inds = torch.nonzero(labels >= 1).squeeze()
    if inds.numel() > 0:
        bin_labels[inds, labels[inds] - 1] = 1
    if label_weights is None:
        bin_label_weights = None
    else:
        bin_label_weights = label_weights.view(-1, 1).expand(
            label_weights.size(0), label_channels)
    return bin_labels, bin_label_weights

def binary_cross_entropy(pred,
                         label,
                         weight=None,
                         reduction='mean',
                         avg_factor=None):
    if pred.dim() != label.dim():
        label, weight = _expand_binary_labels(label, weight, pred.size(-1))

    # weighted element-wise losses
    if weight is not None:
        weight = weight.float()

    loss = F.binary_cross_entropy_with_logits(
        pred, label.float(), weight, reduction='none')
    loss = weight_reduce_loss(loss, reduction=reduction, avg_factor=avg_factor)

    return loss

class ResampleLoss(nn.Module):
    def __init__(self,args): 
        super(ResampleLoss,self).__init__()
        self.loss_weight = args.loss_weight
        self.reduction = args.reduction 
        self.cls_criterion = binary_cross_entropy
        # reweighting function
        self.reweight_func = args.reweight_func

        # focal loss params
        self.focal = args.focal
        self.gamma = args.gamma
        self.balance_param = args.balance_param
    
        # mapping function params
        self.map_alpha = args.alpha
        self.map_beta = args.beta
        self.map_gamma = args.map_param_gamma
    
        self.class_freq = torch.from_numpy(np.asarray(
            mmcv.load(args.freq_file)['class_freq'])).float().cuda()
        self.neg_class_freq = torch.from_numpy(
            np.asarray(mmcv.load(args.freq_file)['neg_class_freq'])).float().cuda()
        self.num_classes = self.class_freq.shape[0]
        self.train_num = self.class_freq[0] + self.neg_class_freq[0]
        logit_reg=dict(
            neg_scale = args.neg_scale,
            init_bias= args.init_bias
        )
        self.logit_reg = logit_reg
        self.neg_scale = logit_reg[
            'neg_scale'] if 'neg_scale' in logit_reg else 1.0
        init_bias = logit_reg['init_bias'] if 'init_bias' in logit_reg else 0.0

        self.init_bias = - torch.log(
            self.train_num / self.class_freq - 1) * init_bias / self.neg_scale
        
        self.freq_inv = torch.ones(self.class_freq.shape).cuda() / self.class_freq
        self.propotion_inv = self.train_num / self.class_freq

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        weight = self.reweight_functions(label)
        cls_score, weight = self.logit_reg_functions(label.float(), cls_score, weight)
        if self.focal:
            logpt = - self.cls_criterion(
                cls_score.clone(), label, weight=None, reduction='none',
                avg_factor=avg_factor)
            # pt is sigmoid(logit) for pos or sigmoid(-logit) for neg
            pt = torch.exp(logpt)
            loss = self.cls_criterion(
                cls_score, label.float(), weight=weight, reduction='none')
            loss = ((1 - pt) ** self.gamma) * loss
            loss = self.balance_param * loss
        else:
            loss = self.cls_criterion(cls_score, label.float(), weight,
                                        reduction=reduction)

        loss = self.loss_weight * loss
        return loss.mean()

    def reweight_functions(self, label):
        weight = self.rebalance_weight(label.float())
        return weight

    def rebalance_weight(self, gt_labels):
        repeat_rate = torch.sum(gt_labels.float() * self.freq_inv, dim=1, keepdim=True)
        pos_weight = self.freq_inv.clone().detach().unsqueeze(0) / repeat_rate
        # pos and neg are equally treated
        weight = torch.sigmoid(self.map_beta * (pos_weight - self.map_gamma)) + self.map_alpha
        return weight

    def logit_reg_functions(self, labels, logits, weight=None):
        if not self.logit_reg:
            return logits, weight
        if 'init_bias' in self.logit_reg:
            logits += self.init_bias
        if 'neg_scale' in self.logit_reg:
            logits = logits * (1 - labels) * self.neg_scale  + logits * labels
            weight = weight / self.neg_scale * (1 - labels) + weight * labels
        return logits, weight

def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())


def order_sim(im, s):
    """Order embeddings similarity measure $max(0, s-im)$
    """
    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
            - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()
    return score

class MMContrastiveLoss(nn.Module):
    """
    Compute Multimodal Contrastive Loss
    """
    def __init__(self, margin=0, measure=False, max_violation=False):
        super(MMContrastiveLoss, self).__init__()
        self.margin = margin
        if measure == 'order':
            self.sim = order_sim
        else:
            self.sim = cosine_sim

        self.max_violation = max_violation

    def forward(self, im, s):
        # compute image-sentence score matrix
        scores = self.sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum(), cost_im.sum()

def focal_binary_cross_entropy(args,p, targets, gamma=2):
    p = p.reshape(-1)
    t = targets.reshape(-1)
    p = torch.where(t >= 0.5, p, 1-p)
    logp = - torch.log(torch.clamp(p, 1e-4, 1-1e-4))
    loss = logp*((1-p)**gamma)
    loss = args.num_classes*loss.mean()
    return loss

def diversity_measurement(supervise_imgpredict, supervise_textpredict, reduce=True):
    div = nn.CosineSimilarity(dim=1)(supervise_imgpredict, supervise_textpredict).mean(axis=0)
    return div

def consistency_measurement(unsupervise_imgpredict, unsupervise_textpredict, cita, reduce=True):

    dis = 2 - nn.CosineSimilarity(dim=1)(unsupervise_imgpredict, unsupervise_textpredict)          

    tensor1 = dis[torch.abs(dis) < cita]
    tensor2 = dis[torch.abs(dis) >= cita]
    tensor1loss = torch.sum(tensor1 * tensor1/2)
    tensor2loss = torch.sum(cita * (torch.abs(tensor2) - 1/2 * cita))

    unsupervise_loss = (tensor1loss + tensor2loss)/unsupervise_imgpredict.size()[0]      
    return unsupervise_loss

# def consistency_measurement_mse(unsupervise_imgpredict, unsupervise_textpredict, reduce=True):

    # dis = 2 - nn.CosineSimilarity(dim=1)(unsupervise_imgpredict, unsupervise_textpredict)          

    # tensor1 = dis[torch.abs(dis) < cita]
    # tensor2 = dis[torch.abs(dis) >= cita]
    # tensor1loss = torch.sum(tensor1 * tensor1/2)
    # tensor2loss = torch.sum(cita * (torch.abs(tensor2) - 1/2 * cita))

    # unsupervise_loss = (tensor1loss + tensor2loss)/unsupervise_imgpredict.size()[0]      
    # return unsupervise_loss



def zlpr_loss(logits,labels):
    pos_labels = labels
    pos_exp_logits = torch.exp(-logits)
    neg_labels = 1-labels
    neg_exp_logits = torch.exp(logits)
    loss = torch.log(1+torch.einsum('bi,bi->b',pos_labels,pos_exp_logits)) 
    + torch.log(1+torch.einsum('bi,bi->b',neg_labels,neg_exp_logits))
    return loss.sum()

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()

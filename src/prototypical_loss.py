# coding=utf-8
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules import Module
from torch.autograd import Variable
from parser_util import get_parser

class BiasLayer(Module):
    '''
    Define Bias Layer to implement Re-measure method
    '''
    def __init__(self):
        super(BiasLayer, self).__init__()
        self.opt = get_parser().parse_args()

        self.alpha = nn.Parameter(torch.ones(self.opt.total_cls, requires_grad=True, device="cuda"))
        self.beta = nn.Parameter(torch.zeros(self.opt.total_cls, requires_grad=True, device="cuda"))
    def forward(self, x):
        '''
        Only the current stage data could be used to train. use torch.cat to achieve.
        
        '''
        x = x.to('cuda')
        start,end = x.size(1)-self.opt.class_per_stage,x.size(1)
        alpha,beta = torch.cat([self.alpha[0:start].detach(),self.alpha[start:end]],dim=0),torch.cat([self.beta[0:start].detach(),self.beta[start:end]],dim=0)
        return alpha.mul(x) + beta
    def printParam(self, i):
        print(i, self.alpha, self.beta)

class PrototypicalLoss(Module):
    '''
    Loss class deriving from Module for the prototypical loss function defined below
    '''
    def __init__(self, n_support):
        super(PrototypicalLoss, self).__init__()
        self.n_support = n_support
    def forward(self, input, target, opt, old_prototypes, inc_i):
        return prototypical_loss(input, target, self.n_support, opt, old_prototypes, inc_i)


def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N_query(n) x D
    # y: N_Classes(m) x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def prototypical_loss(input, target, opt, old_prototypes, inc_i,biasLayer,t_prototypes=None):
    '''
    Compute the barycentres by averaging the features of n_support
    samples for each class in target, computes then the distances from each
    samples' features to each one of the barycentres, computes the
    log_probability for each n_query samples for each one of the current
    classes, of appartaining to a class c, loss and accuracy are then computed
    and returned
    Args:
    - input: the model output for a batch of samples
    - target: ground truth for the above batch of samples
    - n_support: number of samples to keep in account when computing
      barycentres, for each one of the current classes
    '''
    target_cpu = target.to('cpu')
    input_cpu = input.to('cpu')
    def supp_idxs(c):
        # FIXME when torch will support where as np
        return target_cpu.eq(c).nonzero()[:opt.num_support_tr].squeeze(1)

    # FIXME when torch.unique will be available on cuda too
    cn = opt.class_per_stage
    if inc_i is None:
        classes = target_cpu.unique()
    else:
        classes = torch.arange(inc_i*cn,(inc_i+1)*cn)
    n_target = len(target_cpu)
    # FIXME when torch will support where as np
    # assuming n_query, n_target constants
    #n_query = target_cpu.eq(classes[0].item()).sum().item() - n_support
    support_idxs = list(map(supp_idxs, classes))
    #if not old_prototypes is None:
    #    print(old_prototypes.size()[0])
    n_prototypes = None
    if not inc_i is None:
        n_prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs])
        n_prototypes = n_prototypes.where(n_prototypes==n_prototypes,torch.full(n_prototypes.size(),opt.edge))
        #prototypes = torch.cat([old_prototypes,n_prototypes.clone()],dim=0)

    if old_prototypes is None:
        prototypes = n_prototypes
    elif inc_i is None:
        prototypes = old_prototypes
    elif old_prototypes.size()[0]>=(inc_i+1)*opt.class_per_stage:
        prototypes = torch.cat([old_prototypes[:inc_i*opt.class_per_stage],n_prototypes],dim=0)
    elif not old_prototypes is None:
        prototypes = torch.cat([old_prototypes,n_prototypes],dim=0)
    else:
        prototypes = n_prototypes


    n_classes = prototypes.size()[0]

    # FIXME when torch will support where as np
    #for x in classes:

    #query_idlist = list(map(lambda c: target_cpu.eq(c).nonzero(), classes))
    #query_idxs = torch.cat(query_idlist).view(-1)
    #query_samples = torch.stack([input_cpu[query_lists] for query_lists in query_idxs])
    #query_samples = input_cpu[query_idxs]
    n_query = len(input_cpu)
    dists = euclidean_dist(input_cpu, prototypes)
    bic_dists = biasLayer(dists).to('cpu')
    log_p_y = F.log_softmax(-bic_dists, dim=1)
    softmax_dist = torch.ones(dists.size(0),dists.size(1))-F.softmax(dists,dim=1)
    if not n_prototypes is None:
        prototype_center = n_prototypes.mean(0)
        prototype_dist = euclidean_dist(n_prototypes,n_prototypes)
        d = prototype_dist.size(0)
        diagonal = torch.eye(d)
        n_diagonal = diagonal.eq(0).bool()
        prototype_dist = torch.where(prototype_dist==0,torch.full_like(prototype_dist, 0.0001),prototype_dist)
        prototype_center_dist = torch.pow(n_prototypes-prototype_center.unsqueeze(0).expand_as(n_prototypes),2).sum(1)
        prototype_dist_loss = prototype_center_dist.mean().detach().pow(2)/torch.masked_select(prototype_dist,n_diagonal).mean()
        print(prototype_dist)
        print(n_prototypes.size())
        print(prototype_center.size())
        print(prototype_center_dist.mean().pow(2))
        print(torch.masked_select(prototype_dist,n_diagonal).mean())
        prototype_center_loss = torch.pow(F.softmax(prototype_center_dist,dim=0),2).sum()
        c_dist_loss = prototype_dist_loss+prototype_center_loss
    else :
        c_dist_loss = 0

    #target_inds = torch.arange(0, n_query)
    #target_inds = target_inds.view(1, n_query)
    #target_inds = target_inds.expand(n_classes, n_query).long()
    #target_inds = target_inds.eq()
    #prototype_dist = euclidean_dist(prototypes,prototypes)
    
    _, y_hat = log_p_y.max(1)
    #target_inds = torch.arange(0, n_classes).view(n_classes, 1, 1).expand(n_classes, n_query, 1).long()
    #target_inds = Variable(target_inds, requires_grad=False)
    target_inds = torch.zeros(len(target_cpu),n_classes).long()
    
    target_inds = target_inds.scatter_(dim=1, index=target_cpu.unsqueeze(1).long(), src=torch.ones(len(target_cpu), n_classes).long())

    #target_inds = target_inds.transpose(0,1)
    #target_inds = [target_inds.index_put_(query_idl,query_idl) for query_idl in query_idlist]
    target_ninds = target_inds.eq(0)
    
    #proto_dist_mask = prototype_dist.eq(0)
    #proto_dist_mask = proto_dist_mask.eq(0)
    #dist_loss = torch.rsqrt(torch.masked_select(prototype_dist,proto_dist_mask.bool())).mean()
     #+log_p_y.squeeze().view(-1).mean()
    if opt.lossF=='NCM':
        loss_val = c_dist_loss-torch.masked_select(log_p_y,target_inds.bool()).mean()
    else:
        entropy = nn.CrossEntropyLoss()
        loss_val= c_dist_loss+entropy(F.softmax(-dists,dim=1),target_cpu)
    '''if not t_prototypes is None and not n_prototypes is None:
        self_dist = euclidean_dist(n_prototypes,t_prototypes)
        d = self_dist.size(0)
        self_ind = torch.zeros(d,d).long()
        self_ind = self_ind.scatter_(dim=1,index = torch.arange(d).unsqueeze(0).long(), src = torch.ones(d,d).long())
        self_dist_loss = torch.masked_select(F.softmax(self_dist,dim=1),self_ind.bool()).mean()
        loss_val = loss_val+self_dist_loss'''
    #loss_val = -log_p_y.gather(1, target_inds).squeeze().view(-1).mean()
    acc_val = y_hat.eq(target_cpu.squeeze()).float().mean()

    return loss_val,  acc_val, n_prototypes


def com_proto(img_input):
    #input size = n_class x len(image) x d
    return img_input.mean(1)
    n_class = img_input.size(0)
    n = img_input.size(1)
    d = img_input.size(2)
    ori_prototypes = img_input.mean(1)
    dis_factor = F.softmax(torch.rsqrt(torch.pow((ori_prototypes.unsqueeze(1).expand(n_class,n,d)-img_input),2).sum(2)),dim=1)#size = n

    prototypes = img_input.mul(dis_factor.unsqueeze(2).expand(n_class,n,d)).sum(1)
    prototypes = torch.where(prototypes==0,torch.full_like(prototypes, 0.0001),prototypes)
    return prototypes #n_class x d

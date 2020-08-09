# coding=utf-8
from prototypical_batch_sampler import PrototypicalBatchSampler
from prototypical_loss import prototypical_loss as loss_fn
from prototypical_loss import BiasLayer
from prototypical_loss import euclidean_dist
from prototypical_loss import com_proto
from omniglot_dataset import OmniglotDataset
from protonet import ProtoNet
from model import PreResNet
from parser_util import get_parser
from cifar import Cifar100
from torch.utils.data import DataLoader
from torchvision import transforms
from exemplar import Exemplar
from dataset import BatchData
from torchvision.transforms import Compose, CenterCrop, Normalize, Scale, Resize, ToTensor, ToPILImage
from tqdm import tqdm
import numpy as np
import torch
import random
import os
import argparse
import copy
import torch.nn as nn
from torch.nn import functional as F



def init_seed(opt):
    '''
    Disable cudnn to maximize reproducibility
    '''
    torch.cuda.cudnn_enabled = False
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed(opt.manual_seed)


def init_dataset(opt, mode):
    '''
    Initialize dataset
    '''
    total_cls = opt.total_cls
    exemplar = Exemplar(max_size, total_cls)
    dataset = Cifar100()

    #dataset = OmniglotDataset(mode=mode, root=opt.dataset_root)
    #n_classes = len(np.unique(dataset.y))
    #if n_classes < opt.classes_per_it_tr or n_classes < opt.classes_per_it_val:
        #raise(Exception('There are not enough classes in the dataset in order ' +
         #               'to satisfy the chosen classes_per_it. Decrease the ' +
          #              'classes_per_it_{tr/val} option and try again.'))
    return dataset


def init_sampler(opt, labels, mode):
    '''
    Initialize sampler
    '''
    if 'train' in mode:
        classes_per_it = opt.classes_per_it_tr
        num_samples = opt.num_support_tr + opt.num_query_tr
    else:
        classes_per_it = opt.classes_per_it_val
        num_samples = opt.num_support_val + opt.num_query_val

    return PrototypicalBatchSampler(labels=labels,
                                    classes_per_it=classes_per_it,
                                    num_samples=num_samples,
                                    iterations=opt.iterations)


def init_dataloader(opt, mode):
    '''
    Initialize the dataloader
    '''
    dataset = init_dataset(opt, mode)
    sampler = init_sampler(opt, dataset.y, mode)
    dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)
    return dataloader


def init_protonet(opt):
    '''
    Initialize the ProtoNet
    '''
    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'
    model = ProtoNet().to(device)
    return model


def init_optim(opt, model):
    '''
    Initialize optimizer
    '''
    return torch.optim.Adam(params=model.parameters(),
                            lr=opt.learning_rate)


def init_lr_scheduler(opt, optim):
    '''
    Initialize the learning rate scheduler
    '''
    return torch.optim.lr_scheduler.StepLR(optimizer=optim,
                                           gamma=opt.lr_scheduler_gamma,
                                           step_size=opt.lr_scheduler_step)


def save_list_to_file(path, thelist):
    with open(path, 'w') as f:
        for item in thelist:
            f.write("%s\n" % item)

def testf(opt, test_dataloader, model, prototypes, n_per_stage, biasLayer):
    '''
    Test the model trained with the prototypical learning algorithm
    '''
    

    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'
    avg_acc = list()
    tem_acc = list()
    label = list()
    ind = 0
    count=0
    stage_acc = list()
    for epoch in range(10):
        
        for i, (x, y) in enumerate(tqdm(test_dataloader)):
            t = y.squeeze(-1)
            label.extend([ty.item() for ty in t])
            x, y = x.to(device), y.squeeze(-1).to(device)
            model_output = model(x)
            _, acc,_= loss_fn(model_output, target=y,
                              opt=opt, old_prototypes=prototypes,inc_i=None,biasLayer = biasLayer)
            avg_acc.append(acc.item())
            if epoch ==9:
                tem_acc.append(acc.item())
                count = count+t.size(0)
                #print(n_per_stage)
                if ind<len(n_per_stage) and count>=n_per_stage[ind]:
                #print("ind:{}".format(ind))
                    stage_acc.append(np.mean(tem_acc))
                    tem_acc.clear()
                    ind = ind+1
                    count=0

    avg_accm = np.mean(avg_acc)
    print('Test Acc: {}'.format(avg_accm))
    print('Test Acc: {}'.format(list([round(c, 4)] for c in avg_acc[:len(test_dataloader)])))


    return avg_acc


def compute_NCM_img_id(input_cpu,target,n_support, num_support_NCM):
    '''
    Calculate the id of the picture closest to the center of the type in the NCM classification method
    '''

    target_cpu = target
    classes = torch.unique(target_cpu)
    def supp_idxs(c):
        return target_cpu.eq(c).nonzero().squeeze(1)
    def class_img(c):
        return target_cpu.eq(c).nonzero().squeeze(1)
    support_idxs = list(map(supp_idxs, classes))
    
    NCM = torch.zeros([1,num_support_NCM]).long().to('cuda')
    for i,class_index in enumerate(support_idxs):
        n = len(class_index)
        d = input_cpu.size(1)
        img = input_cpu[class_index]
        proto = com_proto(img.unsqueeze(0))
        dis = torch.pow((proto.expand(n,d)-img),2).sum(1)
        ord_dis,index = torch.sort(dis,dim=0,descending=False)
        img_index = class_index[index]
        NCM = torch.cat([NCM,img_index[:num_support_NCM].unsqueeze(0)],dim=0)
    return NCM[1:].to('cpu')
 



def get_mem_tr(support_imgs,num_support_NCM):
    '''
    Extract pictures in memory that can be used for training
    '''
    if support_imgs is None:
        return [],[]
    mem_img = torch.split(support_imgs,num_support_NCM,dim=0) # n_class x n_support_NCM x img.size
    n_c = len(mem_img)
    mem_xs=[]
    mem_ys=[]
    for i in range(n_c):
        for m in mem_img[i]:
            mem_xs.append(np.rollaxis(m.squeeze().numpy(),0,3))
        mem_ys.extend([i]*num_support_NCM)
    return mem_xs,mem_ys


def train(opt, model, optim, lr_scheduler, biasLayer, bisoptim, bias_scheduler):
    '''
    Train the model with the prototypical learning algorithm
    '''
    
    input_transform= Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32,padding=4),
                    ToTensor(),
                    Normalize([0.5071,0.4866,0.4409],[0.2673,0.2564,0.2762])])

    input_transform_eval= Compose([
                        ToTensor(),
                        Normalize([0.5071,0.4866,0.4409],[0.2673,0.2564,0.2762])])

    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'

    total_cls = opt.total_cls
    #exemplar = Exemplar(opt.max_size, total_cls)
    dataset = Cifar100()
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    best_acc = 0

    best_model_path = os.path.join(opt.experiment_root, 'best_model.pth')
    last_model_path = os.path.join(opt.experiment_root, 'last_model.pth')


    test_xs = []
    test_ys = []
    train_xs = []
    train_ys = []
    test_vx = []
    test_vy = []
    test_accs = []
    n_per_stage = []
    support_imgs = None
    prototypes = None
    
    for inc_i in range(opt.stage):
        #exemplar.clear()
        print(f"Incremental num : {inc_i}")
        train, val, test = dataset.getNextClasses(inc_i)
        train_x, train_y = zip(*train)
        val_x, val_y = zip(*val)
        test_x, test_y = zip(*test)
        #print(f"train:{train_y}")
        train_y_hot = dense_to_one_hot(train_y,100)
        val_y = dense_to_one_hot(val_y,100)
        test_y = dense_to_one_hot(test_y,100)
        test_xs.extend(test_x)
        test_ys.extend(test_y)
        train_xs.clear()
        train_ys.clear()
        #print(f"train_y:{train_y} ,val_y:{val_y}, test_y:{test_y}")
        #train_xs, train_ys = exemplar.get_exemplar_train()
        train_xs.extend(train_x)
        train_xs.extend(val_x)
        train_ys.extend(train_y)
        train_ys.extend(val_y)
        train_xNCM = train_xs[:]
        train_yNCM = train_ys[:]

        NCM_dataloader = DataLoader(BatchData(train_xNCM, train_yNCM, input_transform),
                    batch_size=opt.NCM_batch, shuffle=True, drop_last=True)
        mem_xs,mem_ys = get_mem_tr(support_imgs,opt.num_support_NCM)
        train_xs.extend(mem_xs)
        train_ys.extend(mem_ys)
        tr_dataloader = DataLoader(BatchData(train_xs, train_ys, input_transform),
                    batch_size=opt.batch_size, shuffle=True, drop_last=True)
        val_dataloader = DataLoader(BatchData(val_x, val_y, input_transform_eval),
                    batch_size=opt.batch_size, shuffle=False)
        test_data = DataLoader(BatchData(test_xs, test_ys, input_transform_eval),
                    batch_size=opt.batch_size*2, shuffle=True)
        mem_data = DataLoader(BatchData(mem_xs, mem_ys, input_transform_eval),
                    batch_size=256, shuffle=False, drop_last=False)
        #exemplar.update(total_cls//opt.stage, (train_x, train_y), (val_x, val_y))
        n_per_stage.append(len(test_data) if len(n_per_stage)==0 else (len(test_data)-n_per_stage[-1]))
        
        for epoch in range(opt.epochs-2*inc_i):
            print('=== Epoch: {} ==='.format(epoch))
            #tr_iter = iter(tr_dataloader)
            model.train()
            train_acc.clear()
            train_loss.clear()
            #optim.zero_grad()
            t_prototypes=None
            for i, (cx, cy) in enumerate(tqdm(tr_dataloader)):


                optim.zero_grad()
                #print("x:{},y:{}".format(x.size(),y.squeeze().size()))
                x, y = cx.to(device), cy.squeeze().to(device)
                nt = int(y.size(0)/16)
                model_output = model(x)
                loss, acc, n_prototypes= loss_fn(model_output, target=y, opt=opt, 
                    old_prototypes=None if prototypes is None else prototypes.detach(), inc_i=inc_i,biasLayer=biasLayer,t_prototypes=None if t_prototypes is None else t_prototypes.detach())
                loss_distill = 0
                '''
                this section use memory data to compute a center that fix the imbalance of training data

                '''
                if not prototypes is None:
                    for memx,memy in mem_data:
                        dx,dy = memx.to(device), memy.squeeze().to(device)
                        nm = dy.size(0)
                        r = random.random()*0.9+0.1
                        start,end = 0 if r-0.1<0 else int(nm*(r-0.1)), int(r*nm)
                        nm = end-start
                        mx,my = dx[start:end],dy[start:end]
                        imsize1,imsize2,imsize3 = mx.size(1),mx.size(2),mx.size(3)
                        '''
                        Sample Mix
                        If the caller specified mix parameter.
                        '''
                        if opt.mix:
                            mix_mem = mx.unsqueeze(1).expand(nm,nt,imsize1,imsize2,imsize3)
                            mix_tr = x[:nt].unsqueeze(0).expand(nm,nt,imsize1,imsize2,imsize3)
                            mixup = (mix_mem+mix_tr)/2
                            mixup = mixup.view(nm*nt,imsize1,imsize2,imsize3)
                            mixup = torch.cat([mixup,mixup],dim=0)
                            y1 = y[:nt].unsqueeze(0).expand(nm,nt)
                            y1 = y1.contiguous().view(-1)
                            y2 = my.unsqueeze(1).expand(nm,nt)
                            y2 = y2.contiguous().view(-1)
                            mix_y = torch.cat([y1,y2],dim=0)
                            mix_output = model(mixup)
                            mix_loss,_,_ = loss_fn(mix_output,target=mix_y, opt=opt,old_prototypes=None if prototypes is None else torch.cat([prototypes.detach(),n_prototypes],dim=0), inc_i=None,biasLayer=biasLayer,t_prototypes=None if t_prototypes is None else t_prototypes.detach())
                            
                            loss = loss+0.1*mix_loss
                        model_output = model(dx)
                        loss_distill = loss_distill+proto_distill(model_output,dy,prototypes.detach(),opt,n_prototypes,inc_i,opt.centerR)
                        
                loss = loss+loss_distill
                loss.backward()
                optim.step()
                train_loss.append(loss.item())
                train_acc.append(acc.item())
                t_prototypes = n_prototypes
            '''
            This is done in order to make the model train quickly in the previous stages, 
            and gradually reduce epoch in the later stages to avoid over-fitting.
            '''
            if epoch +1== opt.epochs-2*inc_i:
                for x,y in NCM_dataloader:
                    cx,y = x.to(device),y.squeeze().to(device)
                    model_output = model(cx)
                    print("Compute NCM")
                    NCM_img_id = compute_NCM_img_id(model_output,y,opt.num_support_tr,opt.num_support_NCM)#num_support_NCM*stage_per_classes
                    support_img = x.index_select(0,NCM_img_id.view(-1).squeeze())#num_support_NCM*stage_per_classes,img_size
                    break
            avg_loss = np.mean(train_loss)
            avg_acc = np.mean(train_acc)
            
            print('Avg Train Loss: {}, Avg Train Acc: {}'.format(avg_loss, avg_acc))
            lr_scheduler.step()
            #if val_dataloader is None:
                #continue
            model.eval()
            val_acc.clear()
            val_loss.clear()
            for i, (x, y) in enumerate(tqdm(val_dataloader)):
                x, y = x.to(device), y.squeeze().to(device)
                model_output = model(x)
                loss, acc,_= loss_fn(model_output, target=y, opt=opt, old_prototypes=None if prototypes is None else prototypes.detach(),inc_i=inc_i,biasLayer=biasLayer)
                val_loss.append(loss.item())
                val_acc.append(acc.item())
            avg_loss = np.mean(val_loss)
            avg_acc = np.mean(val_acc)
            postfix = ' (Best)' if avg_acc >= best_acc else ' (Best: {})'.format(
                best_acc)
            print('Avg Val Loss: {}, Avg Val Acc: {}{}'.format(
                avg_loss, avg_acc, postfix))
            '''           
            if avg_acc >= best_acc:
                torch.save(model.state_dict(), best_model_path)
                best_acc = avg_acc
                best_state = model.state_dict()
            '''
        
        for epoch in range(opt.Bias_epoch):
            for i, (x, y) in enumerate(tqdm(tr_dataloader)):
                bisoptim.zero_grad()
                x, y = x.to(device), y.squeeze().to(device)
                model_output = model(x)
                loss, acc, _= loss_fn(model_output.detach(), target=y, opt=opt, old_prototypes=None if prototypes is None else prototypes.detach(), inc_i=inc_i,biasLayer=biasLayer)
                loss.backward()
                bisoptim.step()
            bias_scheduler.step()
        #pp = torch.ones([20,256])
        if inc_i ==0:
            support_imgs = support_img
        else:
            #prototypes = torch.ones([20,256])
            #tem = torch.split(support_img,opt.n_support,dim=0)
            support_imgs = torch.cat([support_imgs,support_img],dim=0)#n_classes x n_support x img.size
        
        if not support_imgs is None:
            prototypes = torch.stack(torch.split(model(support_imgs.to(device)),opt.num_support_NCM,dim=0))#n_class x n_support x prototypes.size()--256
            prototypes = prototypes.mean(1).to('cpu')
            print(prototypes.size())
        print('Testing with last model..')
        testf(opt=opt, test_dataloader=test_data, model=model, prototypes=prototypes.to('cpu'), n_per_stage=n_per_stage,biasLayer=biasLayer)
        biasLayer.printParam(0)




def eval(opt):
    '''
    Initialize everything and train
    '''
    options = get_parser().parse_args()

    if torch.cuda.is_available() and not options.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    init_seed(options)
    test_dataloader = init_dataset(options)[-1]
    model = init_protonet(options)
    model_path = os.path.join(opt.experiment_root, 'best_model.pth')
    model.load_state_dict(torch.load(model_path))

    testf(opt=options,
         test_dataloader=test_dataloader,
         model=model)


def main():
    '''
    Initialize everything and train
    '''
    options = get_parser().parse_args()
    if not os.path.exists(options.experiment_root):
        os.makedirs(options.experiment_root)

    if torch.cuda.is_available() and not options.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    init_seed(options)


    #tr_dataloader = init_dataloader(options, 'train')
    #val_dataloader = init_dataloader(options, 'val')
    # trainval_dataloader = init_dataloader(options, 'trainval')
    #test_dataloader = init_dataloader(options, 'test')
    #model = PreResNet(32,options.total_cls).cuda()
    model = init_protonet(options)
    biasLayer = BiasLayer().cuda()
    bisoptim= torch.optim.Adam(biasLayer.parameters(), lr=0.0001)
    bias_scheduler = torch.optim.lr_scheduler.StepLR(bisoptim, step_size=10, gamma=2)
    #model = nn.DataParallel(model, device_ids=[0])
    optim = init_optim(options, model)
    lr_scheduler = init_lr_scheduler(options, optim)
    train(opt=options,
                model=model,
                optim=optim,
                lr_scheduler=lr_scheduler, biasLayer=biasLayer, bisoptim=bisoptim, bias_scheduler=bias_scheduler)
    
    print("----------train finished----------")
    # optim = init_optim(options, model)
    # lr_scheduler = init_lr_scheduler(options, optim)

    # print('Training on train+val set..')
    # train(opt=options,
    #       tr_dataloader=trainval_dataloader,
    #       val_dataloader=None,
    #       model=model,
    #       optim=optim,
    #       lr_scheduler=lr_scheduler)

    # print('Testing final model..')
    # test(opt=options,
    #      test_dataloader=test_dataloader,
    #      model=model)

def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    return labels_dense

def proto_disti(model_output,target,old_prototypes,num_support_tr):
    target_cpu = target.to('cpu')
    input_cpu = model_output.to('cpu')
    def supp_idxs(c):
        return target_cpu.eq(c).nonzero()[:num_support_tr].squeeze(1)
    classes = target_cpu.unique()
    support_idxs = list(map(supp_idxs, classes))
    new_prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs])

    if new_prototypes.size()!=old_prototypes.size():
        print(new_prototypes.size())
        print(old_prototypes.size())
        print(classes)
    T=2
    pre_p = F.softmax(old_prototypes/T,dim=1)
    p = F.log_softmax(new_prototypes/T,dim=1)
    return -torch.mean(torch.sum(pre_p * p, dim=1))*T*T

def proto_distill(model_output,target,old_prototypes,opt,n_prototypes,inc_i,centerR):
    '''
    func proto_distill defines how to reduce the position change of various prototypes 
    in feature space during the training process.

    By using push and pull method
    '''
    target_cpu = target.to('cpu')
    input_cpu = model_output.to('cpu')
    def supp_idxs(c):
        return target_cpu.eq(c).nonzero().squeeze(1)
    classes = target_cpu.unique()
    start,end = classes[0],classes[-1]
    support_idxs = list(map(supp_idxs, classes))
    new_prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs])
    T=2
    R=2
    old_prototypes = old_prototypes[start:end+1]
    if new_prototypes.size()!=old_prototypes.size():
        print(new_prototypes.size())
        print(old_prototypes.size())
        print(classes)
    old_center =  old_prototypes.mean(0) # last prototype center
    n_center = n_prototypes.mean(0)
    max_dis = torch.pow(old_prototypes -old_center.unsqueeze(0).expand_as(old_prototypes),2).sum(1).max()
    n_center_dis = n_prototypes - old_center.unsqueeze(0).expand_as(n_prototypes)
    center_loss = (R*max_dis - torch.pow(old_center - n_center,2).sum()).pow(2).rsqrt() # compute loss function
    pro_dis = euclidean_dist(new_prototypes,old_prototypes)/T
    n_dis = euclidean_dist(n_prototypes,old_prototypes)/T
    n_dis = torch.where(n_dis==0,torch.full_like(n_dis, 0.0001),n_dis)
    pro_dist = torch.where(pro_dis==0,torch.full_like(pro_dis, 0.0001),pro_dis)
    pro_dist = pro_dist/T
    d = pro_dist.size(0)
    self_ind = torch.zeros(d,d).long()
    self_ind = self_ind.scatter_(dim=1,index = torch.arange(d).unsqueeze(1).long(), src = torch.ones(d,d).long())
    #self_nind = self_ind.eq(0)
    #loss_push = torch.masked_select(torch.rsqrt(torch.pow(pro_dist,2)),self_nind.bool()).mean()
    loss_push = torch.rsqrt(n_dis).sum(1).mean()
    loss_pull = torch.masked_select(F.softmax(pro_dis,dim=1),self_ind.bool()).mean()
    print("#######")
    print(center_loss)
    return opt.pullR*(T**inc_i)*loss_pull+opt.pushR/(inc_i+1)*loss_push+centerR*center_loss

if __name__ == '__main__':
    main()

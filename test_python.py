from re import X
from numpy import *
import os
import json
import numpy as np
import torch
import plyer
import random

from sklearn.metrics import roc_auc_score

import torch.distributed as dist
import albumentations

def test20211026():
    l = []
    l.append([1, 2])
    l.append([3, 4])
    m = mat(l)
    print(m)
    print(m.shape)
    print(l)

def test20211102():
    sz1=[0,1,2,3,1]
    ma = zeros((len(sz1), 10))
    print(ma)
    ma[arange(len(sz1)), sz1]=1
    print(ma)

def test20211102_2():
    sz1=random.randn(5,5)
    print(sz1)


#io相关
def test20211110():
    dataset_dir = 'D:/DATASETS/fabric_data/'
    labels_dir = dataset_dir + 'label_json/'

    #框框文件名
    folders = os.listdir(labels_dir)
    jsFiles = []
    for folder in folders:
        t = os.listdir(labels_dir + folder + '/')
        for f in t:
            jsFiles.append(labels_dir + folder + '/' + f)

    #处理json文件

    load_f = open('D:/DATASETS/fabric_data/label_json/1594408408928_dev001/10024282359407_19_2_0.json')
    load_dict = json.load(load_f)
    x0 = load_dict['bbox']['x0']
    print(x0)

def test20211111():
    filedir = 'D:/9709/Desktop/works/fdu_ML/MLpj/一条数据.json' #一条数据  布眼数据集
    jsf = json.load(open(filedir))
    print(len(jsf))
    jsf = jsf[0]
    h = jsf['h']
    t = jsf['t']
    w = jsf['w'] #228
    r = jsf['r'] #228
    l = jsf['l']
    a = jsf['a'] #228
    i = jsf['i'] #228
    id = jsf['id']
    lC = l['C']
    print(len(w))
    print(len(r))
    print(len(a))
    print(len(i))
    print(id)
    return
    xtrain = np.load('D:/9709/Desktop/works/fdu_ML/MLpj/天纺标数据/x_test_PE.npy')
    print(xtrain)
    print(xtrain.shape) #2701,1307   test 676

    ytrain = np.load('D:/9709/Desktop/works/fdu_ML/MLpj/天纺标数据/y_train_PE.npy')
    print(ytrain)
    print(ytrain.shape) #2701,2
    print(dtype(xtrain[0]))

import torch

def test20211112():
    pred = torch.tensor([[0.1,0.9],[0.2,0.8],[0.3,0.7]])
    y = torch.tensor([[0.2,0.8],[0.2,0.8],[0.2,0.8]])
    MAE = torch.mean(torch.abs(pred[:,0]-y[:,0]))
    print(MAE)
    t = pred[:,0]
    print(t)
    print('----------------')
    sz = [1,2,3,4,5,6,7,8,9]
    np.random.shuffle(sz) #Shuffle the data to split
    print(sz)

def test20220125(w, **p):
    print(w)
    print(p)

def test20220129(t):
    w, b = t
    print(w)
    print(b)

def test20220207():
    di={}
    di['a']=1
    di['b']=2
    print(di)
    di=dict(zip(di, map(lambda x: x+1, di.values())))
    print(di)

    import torch.nn.functional as F
    input = torch.randn(3, 5, requires_grad=True)
    target = torch.randint(5, (3,), dtype=torch.int64)
    loss = F.cross_entropy(input, target)
    print(input)
    print(target)

def test20220207_2():
    from fvcore.common.config import CfgNode
    cfg = CfgNode()
    cfg.TRAIN = CfgNode()
    file_name='./default.yaml'
    cfg.merge_from_file(file_name)
    # if args.opts is not None:
    #     cfg.merge_from_list(args.opts)
    print('done')

def label_to_one_hot(x, class_count):
    return torch.eye(class_count)[x.long(),:]

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]

def test20220213():
    output=torch.tensor(
        [
            [0,1,0],
            [1,0,0],
            [0,1,0],
        ]
    )
    target=torch.tensor(
        [
            [0.],
            [1.],
            [1.],
        ]
    )
    acc1=accuracy(output=output, target=target, topk=(1,))
    print(acc1)

def test20220217():

    ls = open('image_list_medium.txt').readlines()
    print(len(ls))
    d=int(len(ls)/12)
    random.shuffle(ls)
    ls1=ls[:d*10]
    ls2=ls[d*10:d*11]
    ls3=ls[d*11:]
    print(len(ls1))
    print(len(ls2))
    print(len(ls3))
    f1=open("ls1.txt","w")
    f1.writelines(ls1)
    f1.close()
    f12=open("ls2.txt","w")
    f12.writelines(ls2)
    f12.close()
    f123=open("ls3.txt","w")
    f123.writelines(ls3)
    f123.close()
    # # ls = open('ls3.txt').readlines()
    # # print(len(ls))
    pass

import torch.nn as nn
class ResNet50(nn.Module):
    def __init__(self, model_cfg) -> None:  # use cfg to set parameters
        super().__init__()
        self.layers = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)
        # self.l2 = 

    def forward(self, samples):
        x = samples['img']
        return self.layers(x)


def test20220222(d):
    d['b']=2

def test20220223():
    pretrained_path='http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth'
    # torch.hub.download_url_to_file(pretrained_path, './tmp/temporary_file')
    state_dict = torch.hub.load_state_dict_from_url(pretrained_path)
    print(state_dict.keys())

import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import auc


def test20220303():
    n=[12,23]
    n2=233
    if isinstance(n, list):
        print(True)
    else:
        print(False)
    t=torch.Tensor(32,37,512)
    t = t[:,-2,:].unsqueeze(2)
    print(t.shape)

def test20220304():
    a=float(input())
    b=float(input())
    c=float(input())
    d=b/((1+a)**c)
    print(d)

def test20220321(t):
    t=t+2


def f1():
    print('f1')

def f2(a:int):
    print('f2', a)

def f3(a:str, b):
    print('f3',a,b)

def test20220324():
    c=dict({'Normalize_ags': [[0.5,0.5,0.5],[0.5,0.5,0.5]]})
    augs=['HorizontalFlip','Normalize']
    t=getattr(albumentations,augs[0])
    t2=albumentations.Compose([t()])
    t2.add_targets()
    pic=torch.Tensor(3,96,96).numpy()
    print(pic.shape)
    pic=t2(image=pic)

def test20220325():
    a=None
    b=2
    if a and b is not None:
        print('yes')

def ttt(a,*p):
    print(len(p))


if __name__ == '__main__':
#     #print('h')
#     # test20211111()

#     # t = (12,3)
#     # test20220129(t)
#     # test20220207()
#     # test20220213()
#     # test20220217()

#     # test20220225()
#     # test20220304()
#     t=111
#     test20220321(id(t))
#     print(t)
#     exit()
    # test20220325()
    a=[1,2,3]
    ttt('as',a)

    exit()
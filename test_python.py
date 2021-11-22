from numpy import *
import os
import json
import numpy as np


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

if __name__ == '__main__':
    #print('h')
    test20211111()
    exit()
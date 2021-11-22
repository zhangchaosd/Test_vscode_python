import torch
import os
from torchvision.io import read_image
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms






def ss(image):
    print(image.dtype)
    plt.axis("off")
    im = image.transpose(0,1)
    im = im.transpose(1,2)
    plt.imshow(im)
    plt.show()


dw = torch.Tensor([[9,-6],[-5,8]])
print(dw)
dw[dw>0]=1
dw[dw<0]=-2
print(dw)
print('hhh')


labels = torch.Tensor([2, 3, 4, 5, 1])
labels2 = np.array([2, 3, 4, 5, 1])

print(len(labels))
print(labels2.shape)

a = torch.Tensor([0,1,2,3,4])
b=a[:-2]
print(b)

img_path = "D:/DATASETS/VOC2012/JPEGImages/2007_000032.jpg"
image = read_image(img_path)
print(image.shape)
image2 = torchvision.transforms.functional.pad(image, padding = (100, 50), fill = 128)
print(image2.shape)
#image2 = torchvision.transforms.functional.resize(image2, (448,448))
print(image2.shape)
h, w = image2.shape[-2:]
print(w,h)
ss(image2)







''''

碧柔洁面泡沫
狮王牙膏
漱口水
护手霜
'''
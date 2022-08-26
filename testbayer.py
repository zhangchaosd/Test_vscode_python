import torch
import math
import random
import cv2
from torchvision.utils import _log_api_usage_once
import numpy as np

class AddGaussianNoise(torch.nn.Module):
    def __init__(self, mean=0., std=1.):
        super().__init__()
        _log_api_usage_once(self)
        self.std = std
        self.mean = mean
        
    def forward(self, img):
        img = img + torch.randn(img.size()) * self.std + self.mean
        return img.byte()
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class AddPoissonNoise(torch.nn.Module):
    def __init__(self, lamb=1.):
        super().__init__()
        _log_api_usage_once(self)
        self.lamb = lamb

    def poisson_value1(self, lamb):
        L = math.exp(-lamb)
        k = 0
        p = 1
        while p >= L:
            k = k + 1
            p = random.random() * p
        return k - 1


    def forward(self, img):
        bs, h, w = img.shape
        for i0 in range(bs):
            for i in range(h):
                for j in range(w):

                    # get the poisson value
                    noise_value = 0
                    for k in range(img[i0][i][j]):
                        noise_value = noise_value + self.poisson_value1(self.lamb)

                    # add noise to original image
                    temp = noise_value
                    if temp > 255:
                        temp = 255

                    # assign noised image to output
                    img[i0][i][j] = temp
        # print(img.dtype)
        return img
    
    def __repr__(self):
        return self.__class__.__name__ + '(lamb={0})'.format(self.lamb)

class AddPoissonNoise2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        _log_api_usage_once(self)

    def forward(self, img):
        return torch.poisson(img/1.).byte()
    
    def __repr__(self):
        return self.__class__.__name__ + '(lamb={0})'.format(self.lamb)


class Bayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        _log_api_usage_once(self)

    def forward(self, img):
        b = np.zeros((img.shape[0], img.shape[2],img.shape[3]), np.uint8)
        h=img.shape[2]
        w=img.shape[3]
        for i in range(h):
            for j in range(w):
                if (i+j)%2==0:
                    b[:,i,j]=img[:,1,i,j]  ####G check
                elif i%2==0:
                    b[:,i,j]=img[:,2,i,j]  ####R
                else:
                    b[:,i,j]=img[:,0,i,j]  ####B
        return torch.from_numpy(b)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"

class Debayer(torch.nn.Module):

    def __init__(self):
        super().__init__()
        _log_api_usage_once(self)

    def forward(self, img):
        bs, h, w = img.shape
        ret=torch.Tensor(bs, 3, h, w)
        for i in range(bs):
            ret[i]=torch.from_numpy(cv2.cvtColor(img[i].numpy(), cv2.COLOR_BAYER_GR2RGB)).permute(2,0,1)
        return ret.byte()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"


input = torch.randn(2,3,256,256)

tr = torch.nn.Sequential(
    Bayer(),
    AddPoissonNoise(),
    # AddGaussianNoise(),
    Debayer(),
)

img = cv2.imread('/Users/zhangchao/Pictures/t.png')
# cv2.imshow("img",img)
# cv2.waitKey()
input = torch.from_numpy(img)
input = input.unsqueeze(0).permute(0,3,1,2)
print(input.shape)
out = tr(input)
print(out.shape)
cv2.imshow("img",out[0].permute(1,2,0).numpy())
cv2.waitKey()
import torch
from torch import nn
from torch.nn import functional as F

class DyConvSelfAtt(nn.Module):
    def __init__(self, input_channel = 3, q_channel = 8, v_channel = 16):
        super().__init__()
        self.q_conv_params = [(nn.Parameter(torch.randn(q_channel, input_channel, 7, 7)),nn.Parameter(torch.randn(q_channel))) for _ in range(input_channel)]  #TODO change
        self.k_conv_paramss = [[(nn.Parameter(torch.randn(q_channel, input_channel, 7, 7)),nn.Parameter(torch.randn(q_channel))) for _ in range(v_channel)] for i in range(input_channel)]  #TODO
        self.v_conv_param = nn.Parameter(torch.randn(v_channel, input_channel, 7, 7)), nn.Parameter(torch.randn(v_channel))  #TODO

    def forward(self, x):
        # bs,c,h,w=x.shape

        # TODO generate params dynamicly

        # get q
        q = torch.stack([F.conv2d(input=x, weight=kernel, bias=bias, stride=2, padding=3) for kernel, bias in self.q_conv_params], dim=1)  #TODO stride, padding,check dim 效果
        # q: bs5, input_channel3, q_channel8, 19, 19

        # get ks
        ks = torch.stack([torch.stack([F.conv2d(input=x, weight=kernel, bias=bias, stride=2, padding=3) for kernel, bias in k_conv_params], dim=1) for k_conv_params in self.k_conv_paramss],dim=1)  #check dim
        # ks: bs5,in_channel3, v_channel16,q_channel8,19,19

        # get as
        a = torch.stack([torch.stack([F.conv2d(input=qii, weight=kii, stride=2, padding=3) for qii, kii in zip(qi,ki)], dim=0) for qi, ki in zip(q, ks)],dim=0)
        # print(a.shape)
        # bs5,in_channel3,v_channel16,4,4

        # get v
        v = F.conv2d(input=x,weight=self.v_conv_param[0],bias=self.v_conv_param[1],stride=2,padding=3)
        # bs5, v_channel16, 19, 19

        # get o
        o=torch.stack([F.conv2d(input=i,weight=w,stride=2,padding=3) for i,w in zip(v,a)], dim=0)
        # bs5, in_channel3, ho11, wo11
        return o


class MultiHeadDyConvSelfAtt(nn.Module):
    def __init__(self, num_head = 4, input_channel = 3, q_channel = 8, v_channel = 16):
        super().__init__()
        self.heads=[DyConvSelfAtt(input_channel=input_channel,q_channel=q_channel,v_channel=v_channel) for _ in range(num_head)]
        self.conv_trans_param = nn.Parameter(torch.randn(input_channel * num_head, input_channel, 7, 7)), nn.Parameter(torch.randn(input_channel))  ## TODO get kernel size

    def forward(self, x):
        print('x input MultiHeadDyConvSelfAtt:', x.shape)
        x = [head(x) for head in self.heads]
        x = torch.cat(x, dim=1)
        x = F.conv_transpose2d(input=x,weight=self.conv_trans_param[0],bias=self.conv_trans_param[1],stride=3,output_padding=1)  #TODO get out pading and stride
        print('x output MultiHeadDyConvSelfAtt:', x.shape)
        return x

class Encoder(nn.Module):
    def __init__(self, num_head = 4, input_channel = 3, q_channel = 8, v_channel = 16):
        super().__init__()
        self.multi_head = MultiHeadDyConvSelfAtt()

    def forward(self, x):
        print('x input Encoder:', x.shape)
        bs, c, h, w = x.shape
        x = F.layer_norm(x+self.multi_head(x), [c, h, w])
        print('x output Encoder:', x.shape)
        return x

def main():
    x = torch.rand(5, 3, 38, 38)
    model = MultiHeadDyConvSelfAtt()
    model.eval()
    o=model(x)
    # print(o.shape)

if __name__ =='__main__':
    main()

    # encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
    # src = torch.rand(10, 32, 512)
    # print(src.shape)
    # out = encoder_layer(src)
    # print(out.shape)

    # def fun(x,y=0):
    #     print(x,y)
    # for each in map(fun, list(zip([1,2,6],[3,4,9]))):
    #     print(each)


    # tezheng i:c
    # wq: c,n1,cgroup
    # wk: c,n1,n2group,cgroup
    # wv: c,n2
    # tezheng q: n1,
    # k: n1,n2,cgroup
    # tezheng v:n2
    # a: n2,c
    # o: c

    #
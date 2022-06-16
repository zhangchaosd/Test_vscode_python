import torch
from torchsummary import summary
import torchvision as tv

print('a')

vgg = tv.models.vgg16(pretrained=False)

summary(vgg,(3,255,255))


####下面用于 PyDeepfakeDet

from PyDeepFakeDet import models


from torchsummary import summary





def ps(model,s):
    cfg = {"PRETRAINED": False, "ESCAPE": "",'VARIANT':'efficientnet-b4'}
    net = getattr(models,model)(cfg)
    summary(net,(3,s,s))


# ps("Xception",299)
# ps("EfficientNet",380)
# ps("Meso4",256)
ps("GramNet",299)
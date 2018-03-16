import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import itertools
import torchvision.models as m
from torchvision.models import resnet, densenet, vgg


import models.common as c
import tools.model.io as io

from tools import Struct


def make_encoder(name, depth = None):
    layers = get_layers(name)
    if depth:
        layers = layers[:depth]
        
    return make_cascade(layers)


def get_layers(name):
    model = m.__dict__[name](pretrained=True)
    
    if isinstance(model, resnet.ResNet):
        return resnet_layers(model)
    elif isinstance(model, densenet.DenseNet):
        return densenet_layers(model)
    elif isinstance(model, vgg.VGG):
        return vgg_layers(model)
    else:
        assert false, "unsupported model type " + name

    

def make_cascade(layers):
    return c.Cascade(*layers)

def layer_sizes(layers):
    return encoder_sizes(make_cascade(layers))

def encoder_sizes(encoder):
    encoder.eval()

    x = Variable(torch.FloatTensor(8, 3, 224, 224))
    skips = encoder(x)

    return [t.size(1) for t in skips]

def resnet_layers(resnet):
    layer0 = nn.Sequential(resnet.conv1, resnet.bn1, nn.ReLU(inplace=True))
    layer1 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2, padding=1), resnet.layer1)

    layers = [c.Identity(), layer0, layer1, resnet.layer2, resnet.layer3, resnet.layer4]
    return layers


def vgg_layers(model):
    layers = []
    current = nn.Sequential()

    for i, m in enumerate(model.features._modules.values()):
        if isinstance(m, nn.MaxPool2d):
            layers.append(current)
            current = nn.Sequential(m)
        else:
            current.add_module(str(i), m)

    return layers

def densenet_layers(densenet):
    m = densenet.features._modules

    return [
            c.Identity(),
            nn.Sequential(m['conv0'], m['norm0'], m['relu0']),
            nn.Sequential(m['pool0'], m['denseblock1']),
            nn.Sequential(m['transition1'], m['denseblock2']),
            nn.Sequential(m['transition2'], m['denseblock3']),
            nn.Sequential(m['transition3'], m['denseblock4'], m['norm5'])
        ]




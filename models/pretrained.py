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


def make_encoder(name, depth):
    model = m.__dict__[name](pretrained=True)
    if isinstance(model, resnet.ResNet):
        return resnet_encoder(model, depth)
    elif isinstance(model, densenet.DenseNet):
        return densenet_encoder(model, depth)
    elif isinstance(model, vgg.VGG):
        return vgg_encoder(model, depth)
    else:
        assert false, "unsupported model type " + name


def make_cascade(layers, depth):
    return c.Cascade(*layers[:depth])

def encoder_sizes(encoder):
    encoder.eval()

    x = Variable(torch.FloatTensor(8, 3, 224, 224))
    skips = encoder(x)

    return [t.size(1) for t in skips]

def resnet_encoder(resnet, depth):
    layer0 = nn.Sequential(resnet.conv1, resnet.bn1, nn.ReLU(inplace=True))
    layer1 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2, padding=1), resnet.layer1)

    layers = [c.Identity(), layer0, layer1, resnet.layer2, resnet.layer3, resnet.layer4]
    return make_cascade(layers, depth)


def vgg_encoder(model, depth):
    layers = []
    current = nn.Sequential()

    for i, m in enumerate(model.features._modules.values()):
        if isinstance(m, nn.MaxPool2d):
            layers.append(current)
            current = nn.Sequential(m)
        else:
            current.add_module(str(i), m)

    return make_cascade(layers, depth)

def densenet_encoder(densenet, depth):
    m = densenet.features._modules

    layers = [
            c.Identity(),
            nn.Sequential(m['conv0'], m['norm0'], m['relu0']),
            nn.Sequential(m['pool0'], m['denseblock1']),
            nn.Sequential(m['transition1'], m['denseblock2']),
            nn.Sequential(m['transition2'], m['denseblock3']),
            nn.Sequential(m['transition3'], m['denseblock4'], m['norm5'])
        ]

    return make_cascade(layers, depth)




class EncoderDecoder(nn.Module):

    def __init__(self, args, encoder, num_classes=2):
        super().__init__()
        self.encoder = encoder
        self.modifier = args.lr_modifier

        sizes = encoder_sizes(encoder)
        print(sizes)

        block = c.basic_block(p=args.dropout)
        layers = [args.block_layers] * len(sizes)
        classifier = c.Conv(args.base, num_classes)
        self.decoder = nn.Sequential(c.decoder(args.base, args.inc, sizes, layers, block=block), classifier)
        c.init_weights(self.decoder)

    def forward(self, input):

        skips = self.encoder(input)
        return self.decoder(skips)


    def parameter_groups(self, args):
        return [
                {'params': self.encoder.parameters(), 'lr': self.modifier * args.lr, 'modifier': self.modifier},
                {'params': self.decoder.parameters()}
            ]


parameters = Struct(
        base = (16,    "feature size at top layer"),
        inc  = (16,    "features added per layer"),

        block_layers = (1,   "number of layers at each resolution"),
        base_name    = ("resnet18", "name of pretrained resnet to use"),
        dropout      = (0.0, "dropout after each decoder layer"),
        lr_modifier  = (0.1, "learning rate modifier for pretrained part (fine tuning)"),
        depth        = (6, "depth of network")
    )

extra = Struct(
        levels = (2, "levels of multi scale used"),
        total_depth = (8, "restrict total depth of network")
    )


def create_pretrained(args, num_classes=2, input_channels=3):
    assert input_channels == 3
    encoder = make_encoder(args.base_name, args.depth)
    return EncoderDecoder(args, encoder, num_classes=num_classes)


def create_cascade(args, num_classes=2, input_channels=3):
    assert input_channels == 3
    encoder = make_encoder(args.base_name, args.depth)
    encoder = c.MultiCascade(encoder, depth=args.total_depth, levels=args.levels)

    return EncoderDecoder(args, encoder, num_classes=num_classes)

models = {
    'pretrained' : Struct(create=create_pretrained, parameters=parameters),
    'cascade'    : Struct(create=create_cascade, parameters=parameters + extra)
  }

if __name__ == '__main__':

    _, *cmd_args = sys.argv
    args = io.parse_params(models, cmd_args)

    model = create_cascade(args, 2, 3)

    x = Variable(torch.FloatTensor(4, 3, 500, 500))
    y = model.cuda()(x.cuda())

    print(y.size())

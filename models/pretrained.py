import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import itertools
import torchvision.models as m
from torchvision.models import resnet, densenet


import models.common as c
import tools.model.io as io

from tools import Struct


def make_encoder(name, depth):
    model = m.__dict__[name](pretrained=True)
    if isinstance(model, resnet.ResNet):
        return resnet_encoder(model, depth)
    elif isinstance(model, densenet.DenseNet):
        return densenet_encoder(model, depth)
    else:
        assert false, "unsupported model type " + name


def make_cascade_sizes(layers, depth):
    encoder = c.Cascade(*layers[:depth])
    encoder.eval()

    x = Variable(torch.FloatTensor(8, 3, 224, 224))
    skips = encoder(x)

    return encoder, [t.size(1) for t in skips]

def resnet_encoder(resnet, depth):
    layer0 = nn.Sequential(resnet.conv1, resnet.bn1, nn.ReLU(inplace=True))
    layer1 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2, padding=1), resnet.layer1)

    layers = [c.Identity(), layer0, layer1, resnet.layer2, resnet.layer3, resnet.layer4]
    return make_cascade_sizes(layers, depth)


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

    return make_cascade_sizes(layers, depth)

class Pretrained(nn.Module):

    def __init__(self, args, num_classes=2):
        super().__init__()
        self.encoder, encoder_sizes = make_encoder(args.base_name, args.depth)
        self.modifier = args.lr_modifier

        classifier = c.Conv(args.base, num_classes)

        l = args.block_layers
        layers = [1] * args.depth

        block = c.basic_block(p=0)
        self.decoder = nn.Sequential(c.decoder(args.base, args.inc, encoder_sizes, layers, block=block), classifier)
        c.init_weights(self.decoder)

        self.drop = c.dropout(p=args.dropout)

    def forward(self, input):

        skips = list(map(self.drop, self.encoder(input)))
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
        lr_modifier  = (1.0, "learning rate modifier for pretrained part (fine tuning)"),
        depth        = (6, "depth of network")
    )

def create(args, num_classes=2, input_channels=3):
    assert input_channels == 3
    return Pretrained(args, num_classes=num_classes)

models = {'pretrained' : Struct(create=create, parameters=parameters)}

if __name__ == '__main__':

    _, cmd_args = sys.argv
    args = io.parse_params(models, [cmd_args])

    model = create(args, 2, 3)

    x = Variable(torch.FloatTensor(4, 3, 500, 500))
    y = model.cuda()(x.cuda())

    print(y.size())

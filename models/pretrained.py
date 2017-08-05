import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import itertools
import torchvision.models as m
from torchvision.models import resnet

import models.common as c
import tools.model.io as io

from tools import Struct



def linear_to_conv(m):
    out, inp = m.weight.size()
    conv = nn.Conv2d(inp, out, 1)

    conv.weight.data.copy_(m.weight.data)
    conv.bias.data.copy_(m.bias.data)
    return conv


def make_encoder(name):
    model = m.__dict__[name](pretrained=True)
    if isinstance(model, resnet.ResNet):
        return resnet_encoder(model)
    else:
        assert false, "unsupported model type " + name


def resnet_encoder(resnet):
    layer0 = nn.Sequential(resnet.conv1, resnet.bn1, nn.ReLU(inplace=True))
    layer1 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2, padding=1), resnet.layer1)

    encoder = c.Cascade(layer0, layer1, resnet.layer2, resnet.layer3, resnet.layer4)
    encoder.eval()

    x = Variable(torch.FloatTensor(8, 3, 224, 224))
    skips = encoder(x)

    return encoder, [t.size(1) for t in skips]


class Pretrained(nn.Module):

    def __init__(self, args, num_classes=2):
        super().__init__()
        self.encoder, sizes = make_encoder(args.base_name)

        classifier = c.Conv(args.features, num_classes, 1)
        self.decoder = nn.Sequential(c.Decoder(sizes, args.features, num_blocks=args.block_layers), classifier)

    def forward(self, input):
        return self.decoder(self.encoder(input))


    def parameter_groups(self, args):
        return [
                {'params': self.encoder.parameters(), 'lr':0.1 * args.lr},
                {'params': self.decoder.parameters()}
            ]


parameters = Struct(
        features   = (16,    "hidden feature size"),
        block_layers = (2,   "number of layers at each resolution"),
        base_name  = ("resnet34", "name of pretrained resnet to use"),
        dropout = (0, "dropout after each decoder layer")
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

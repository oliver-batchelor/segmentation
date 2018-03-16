import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import itertools
import torchvision.models as m
from torchvision.models import resnet, densenet, vgg

import models.pretrained as pretrained

import models.common as c
import tools.model.io as io

from tools import Struct



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
        extra_features = (512, "number of features in extra layers"),
        dropout      = (0.0, "dropout after each decoder layer"),
        lr_modifier  = (0.1, "learning rate modifier for pretrained part (fine tuning)"),
        depth        = (6, "depth of network")
    )

def extra_layer(features):
    convs = nn.Sequential (Conv(features, features), Conv(features, features))
    return nn.Sequential(Residual(convs), Conv(features, features, stride=2))

def create_pretrained(args, num_classes=2, input_channels=3):
    assert input_channels == 3
    encoder = pretrained.make_encoder(args.base_name, args.depth)
    return EncoderDecoder(args, encoder, num_classes=num_classes)


models = {
    'pretrained' : Struct(create=create_pretrained, parameters=parameters)
  }

if __name__ == '__main__':

    _, *cmd_args = sys.argv
    args = io.parse_params(models, cmd_args)

    model = create_pretrained(args, 2, 3)

    x = Variable(torch.FloatTensor(4, 3, 500, 500))
    y = model.cuda()(x.cuda())

    print(y.size())

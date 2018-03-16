import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import models.common as c
import tools.model.io as io

from tools import Struct



class Ladder(nn.Module):

    def __init__(self, args, num_classes=2, input_channels=3):
        super().__init__()

        classifier = c.Conv(args.base, num_classes)
        self.initial = nn.Conv2d(input_channels, args.base, 3, padding=1)

        l = args.block_layers
        layers = [1 * l, 2 * l] + [3 * l] * (args.depth - 2)

        block = c.basic_block(p=args.dropout)
        self.encoder, encoder_sizes = c.encoder(args.base, args.inc, layers, block=block)
        self.decoder = nn.Sequential(c.decoder(args.base, args.inc, encoder_sizes, layers, block=block), classifier)
        c.init_weights(self)

#        print(encoder_sizes)

    def forward(self, input):
        input = self.encoder(self.initial(input))
#        print([x.size() for x in input])

        return self.decoder(input)


    def parameter_groups(self, args):
        return [{'params': self.parameters()}]


parameters = Struct(
        base = (16,    "feature size at top layer"),
        inc  = (16,    "features added per layer"),

        block_layers = (1,   "number of layers at each resolution"),
        depth        = (6,   "maximum depth of the network"),
        dropout = (0.0, "dropout after each decoder layer")
    )

def create(args, num_classes=2, input_channels=3):
    return Ladder(args, num_classes=num_classes, input_channels=input_channels)

models = {'ladder' : Struct(create=create, parameters=parameters)}

if __name__ == '__main__':

    _, cmd_args = sys.argv
    args = io.parse_params(models, [cmd_args])

    model = create(args, 2, 3)

    x = Variable(torch.FloatTensor(4, 3, 500, 500))
    y = model.cuda()(x.cuda())

    print(y.size())



import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn.init as init

from tools import Struct
from tools.model import match_size_2d

import models.pyramid as pyramid
import math

parameters = Struct(
        depth      = (8,    "number of layers of depth in the model"),
        features   = (16,    "hidden feature size"),
        growth     = (1.5, "feature growth between layers"),
        dropout    = (0.05,  "dropout level between convolutions"),
        leaky      = (0.2,  "slope parameter to use for leaky-relu"),
        kernel     = (5, "size of kernels")
    )

def create(args, num_classes=2, input_channels=3):
    def identity(x):
        return x

    class Conv(nn.Module):

        def __init__(self, in_size, out_size, kernel=args.kernel, stride = 1, padding=None):
            super().__init__()

            self.conv = nn.Conv2d(in_size, out_size, kernel, stride=stride, padding=padding or (kernel//2))

            self.norm =  nn.BatchNorm2d(out_size)
            self.drop = nn.Dropout2d(p=args.dropout)

        def forward(self, inputs):
            return self.drop(F.leaky_relu(self.norm(self.conv(inputs)), args.leaky, inplace=True))


    class Encode(nn.Module):

        def __init__(self, in_size, out_size):
            super().__init__()
            self.conv = Conv(in_size, out_size)

            self.skip_size = out_size

        def forward(self, inputs):
            output = self.conv(inputs)
            return F.max_pool2d(output, 2, 2, ceil_mode = True), output


    class Decode(nn.Module):

        def __init__(self, in_size, skip_size, out_size):
            super().__init__()

            self.conv = Conv(out_size + skip_size, out_size)
            self.up = nn.ConvTranspose2d(in_size, out_size, 2, 2)

        def forward(self, inputs, skip):
            upscaled = self.up(inputs)
            padded = match_size_2d(upscaled, skip)

            return self.conv(torch.cat([padded, skip], 1))


    segmenter = pyramid.segmenter(Encode, Decode, Conv, args.growth)

    model = segmenter(input_channels=input_channels, output_channels=num_classes)


    def init_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            init.xavier_uniform(m.weight.data, gain=math.sqrt(2))
            init.constant(m.bias.data, 0.1)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    model.apply(init_weights)
    return model

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn.init as init

from tools import Struct
from tools.model import match_size_2d

import models.pyramid as pyramid
import math

parameters = Struct(
        depth      = (4,    "number of layers of depth in the model"),
        features   = (8,    "hidden feature size"),
        dropout    = (0.1,  "dropout level between convolutions")
    )

# class Conv2(nn.Module):
#     def __init__(self, in_size, out_size, kernel = 3):
#         super().__init__()
#
#         self.conv1 = Conv(in_size, out_size, kernel = kernel, dilation = 1)
#         self.conv2 = Conv(out_size, out_size, kernel = kernel, dilation = 1)
#
#
#     def forward(self, inputs):
#         return self.conv2(self.conv1(inputs))


def create(args, num_classes=2, input_channels=3):
    def identity(x):
        return x

    class Conv(nn.Module):

        def __init__(self, in_size, out_size, kernel = 3, stride = 1, padding=None):
            super().__init__()


            self.conv = nn.Conv2d(in_size, out_size, kernel, stride=stride, padding=padding or (kernel//2))

            self.norm =  nn.BatchNorm2d(out_size)
            self.drop = nn.Dropout2d(p=args.dropout)

        def forward(self, inputs):
            return self.drop(F.leaky_relu(self.norm(self.conv(inputs)), 0.2))

    class Conv2(nn.Module):

        def __init__(self, in_size, out_size):
            super().__init__()
            self.conv1 = Conv(in_size, out_size)
            self.conv2 = Conv(out_size, out_size, dilation=2)

        def forward(self, inputs):
            return self.conv2(self.conv1(inputs))

    class Encode(nn.Module):

        def __init__(self, in_size, out_size):
            super().__init__()
            self.conv = Conv(in_size, out_size, stride=2, padding=0)


        def forward(self, inputs):
            print(inputs.size(), self.conv)
            return self.conv(inputs), inputs
            #output = self.conv(inputs)
            #return F.max_pool2d(output, 2, 2, ceil_mode = True), output


    class Decode(nn.Module):

        def __init__(self, in_size, out_size):
            super().__init__()

            self.conv = Conv(out_size * 2, out_size)
            self.up = nn.ConvTranspose2d(in_size, out_size, 2, 2)

        def forward(self, inputs, skip):

            upscaled = self.up(inputs)
            padded = match_size_2d(upscaled, skip)

            print(upscaled.size(), skip.size(), self.conv)

            return self.conv(torch.cat([padded, skip], 1))




    segmenter =  pyramid.segmenter(Encode, Decode)
    model = segmenter(num_classes=num_classes, input_channels=input_channels)

    def init_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            init.xavier_uniform(m.weight.data, gain=math.sqrt(2))
            init.constant(m.bias.data, 0.1)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    model.apply(init_weights)
    return model

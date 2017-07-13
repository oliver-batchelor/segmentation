import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn.init as init

from tools import Struct
from tools.model import match_size_2d

import models.pyramid as pyramid
import math

parameters = Struct(
        depth      = (5,    "number of layers of depth in the model"),
        features   = (16,    "hidden feature size"),
        growth     = (2.0, "feature growth between layers"),
        dropout    = (0.05,  "dropout level between convolutions"),
        leaky      = (0.2,  "slope parameter to use for leaky-relu"),
        kernel     = (5,    "size of kernels"),
        residual   = (False, "use residual skip connection"),
        batch_momentum = (0.9, "momentum for batch normalization")
    )




def create(args, num_classes=2, input_channels=3):
    def identity(x):
        return x

    def dropout():
        return nn.Dropout2d(p=args.dropout) if args.dropout > 0 else identity

    def threshold(inputs):
        return F.leaky_relu(inputs, args.leaky, inplace=True)

    class Conv(nn.Module):
        def __init__(self, in_size, out_size, kernel=args.kernel, stride=1, padding=None):
            super().__init__()

            self.conv = nn.Conv2d(in_size, out_size, kernel, stride=stride, padding=padding or (kernel//2))
            self.norm =  nn.BatchNorm2d(out_size, momentum=args.batch_momentum)


        def forward(self, inputs):
            return threshold(self.norm(self.conv(inputs)))

    # class Conv(nn.Module):
    #     def __init__(self, in_size, out_size, kernel=3, stride=1, padding=None):
    #         super().__init__()
    #
    #         self.conv = nn.Conv2d(in_size, out_size, kernel, stride=stride, padding=padding or (kernel//2))
    #
    #     def forward(self, inputs):
    #         return F.elu(self.conv(inputs), inplace=True)



    class Encode(nn.Module):

        def __init__(self, in_size, out_size):
            super().__init__()

            self.conv1 = Conv(in_size, out_size)
            self.conv2 = Conv(out_size, out_size)

            self.drop = dropout()
            self.skip_size = out_size



        def forward(self, inputs):
            output = self.drop(self.conv2(self.conv1(inputs)))
            return F.max_pool2d(output, 2, 2), output


    #
    # class Decode(nn.Module):
    #
    #     def __init__(self, in_size, skip_size, out_size):
    #         super().__init__()
    #
    #         self.conv = Conv(out_size + skip_size, out_size)
    #         self.up = nn.ConvTranspose2d(in_size, out_size, 2, 2)
    #
    #         self.drop = nn.Dropout2d(p=args.dropout)
    #
    #
    #     def forward(self, inputs, skip):
    #         upscaled = self.up(inputs)
    #         padded = match_size_2d(upscaled, skip)
    #
    #         return self.drop(self.conv(torch.cat([padded, skip], 1)))

    class Residual(nn.Module):
        def __init__(self, features):
            super().__init__()
            self.conv1 = Conv(features, features, kernel=3)
            self.conv2 = Conv(features, features, kernel=3)

        def forward(self, inputs):
            r = self.conv2(self.conv1(inputs))
            return inputs + r

    class Decode(nn.Module):

        def __init__(self, in_size, skip_size, out_size):
            super().__init__()

            self.conv1 = Conv(skip_size + in_size, out_size)
            self.conv2 = Conv(out_size, out_size)

            self.residual = Residual(skip_size) if args.residual else identity

            self.drop = dropout()


        def forward(self, inputs, skip):

            upscaled = F.upsample_nearest(inputs, scale_factor=2)
            padded = match_size_2d(upscaled, skip)

            skip = self.residual(skip)

            inputs = torch.cat([padded, skip], 1)
            return self.drop(self.conv2(self.conv1(inputs)))


    segmenter = pyramid.segmenter(Encode, Decode, Conv, args.growth)
    model = segmenter(input_channels=input_channels, output_channels=num_classes, features=args.features)


    def init_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            init.xavier_uniform(m.weight.data, gain=math.sqrt(2))
            init.constant(m.bias.data, 1)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    model.apply(init_weights)
    return model

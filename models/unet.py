import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn.init as init

from tools import Struct
from tools.model import match_size_2d,  centre_crop

from models.pyramid import pyramid
import math

parameters = Struct(
        depth      = (6,    "number of layers of depth in the model"),

        base  = (32,    "hidden feature size"),
        inc   = (0,    "hidden feature size"),

        dropout    = (0.0,  "dropout level between convolutions"),
        kernel     = (3,    "size of kernels"),
        skip_convs = (False,    "residual convolutions in skip layer")
    )





def create(args, num_classes=2, input_channels=3):
    def identity(x):
        return x

    def dropout():
        return nn.Dropout2d(p=args.dropout) if args.dropout > 0 else identity

    def threshold(inputs):
        return F.relu(inputs, inplace=True)

    class Conv(nn.Module):
        def __init__(self, in_size, out_size, kernel=args.kernel, stride=1, padding=None):
            super().__init__()

            self.conv = nn.Conv2d(in_size, out_size, kernel, stride=stride, padding=padding or (kernel//2))
            self.norm = nn.BatchNorm2d(out_size)

        def forward(self, inputs):
            return threshold(self.norm(self.conv(inputs)))



    class Encode(nn.Module):

        def __init__(self, in_size, out_size):
            super().__init__()

            self.conv1 = Conv(in_size, out_size)
            self.conv2 = Conv(out_size, out_size, stride=1)

            self.drop = dropout()
            self.skip_size = out_size



        def forward(self, inputs):
            x = self.conv1(inputs)
            output = self.drop(self.conv2(x))
            return F.max_pool2d(output, 2, 2), output



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

            self.residual = Residual(skip_size) if args.skip_convs else identity
            self.drop = dropout()


        def forward(self, inputs, skip):

            upscaled = F.upsample(inputs, scale_factor=2)
            upscaled = match_size_2d(upscaled, skip)

            skip = self.residual(skip)
            inputs = torch.cat([upscaled, skip], 1)
            return self.drop(self.conv2(self.conv1(inputs)))

    Pyramid = pyramid(Encode, Decode, args.inc)

    class Segmenter(nn.Module):

        def __init__(self):
            super().__init__()

            self.conv1 = Conv(input_channels, args.base)
            self.conv2 = nn.Conv2d(args.base, num_classes, 3, padding=1)

            self.pyramid = Pyramid(args.base, args.depth)

        def forward(self, input):
            output = self.conv1(input)
            output = self.conv2(self.pyramid(output))
            return output

        def parameter_groups(self, args):
            return [
                    {'params': self.parameters()}
                ]


    def init_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    model = Segmenter()
    model.apply(init_weights)
    return model


models = {'unet' : Struct(create=create, parameters=parameters)}

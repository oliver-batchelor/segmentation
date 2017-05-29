import torch
import torch.nn as nn
import torch.nn.functional as F

import tools.model.pyramid as pyramid
from tools.model import match_size_2d

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


class Conv(nn.Module):

    def __init__(self, in_size, out_size, kernel = 3, dilation = 1):
        super().__init__()

        self.norm =  nn.BatchNorm2d(in_size)
        self.conv1 = nn.Conv2d(in_size, out_size, kernel, dilation = dilation)

        self.drop = nn.Dropout2d(p=0.2)

        p = (kernel//2) * dilation
        self.padding = (p, p, p, p)

    def forward(self, inputs):
        return F.relu(self.drop(self.conv1(F.pad(self.norm(inputs), self.padding, mode='replicate'))))


class Encode(nn.Module):

    def __init__(self, in_size, out_size):
        super().__init__()
        self.conv = Conv(in_size, out_size)

    def forward(self, inputs):
        output = self.conv(inputs)
        return F.max_pool2d(output, 2, 2, ceil_mode = True), output


class Decode(nn.Module):

    def __init__(self, in_size, out_size):
        super().__init__()

        self.conv = Conv(in_size + out_size, out_size)
        self.up = nn.ConvTranspose2d(in_size, out_size, 2, 2)

    def forward(self, inputs, skip):
        upscaled = self.up(inputs)
        padded = match_size_2d(upscaled, skip)

        return self.conv(torch.cat([padded, skip], 1))


segmenter = pyramid.segmenter(Encode, Decode)

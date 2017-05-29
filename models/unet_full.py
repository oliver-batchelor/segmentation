import torch
import torch.nn as nn
import torch.nn.functional as F

import tools.model.pyramid as pyramid
from tools.model import match_size_2d





class Conv(nn.Module):

    def __init__(self, in_size, out_size, kernel = 3, dilation = 1):
        super().__init__()

        self.drop = nn.Dropout2d(p=0.1)


        self.norm = nn.BatchNorm2d(in_size)
        self.conv1 = nn.Conv2d(in_size, out_size, kernel, dilation = dilation)


    def forward(self, inputs):
        return F.relu(self.drop(self.conv1(self.norm(inputs))))

class ConvT(nn.Module):

    def __init__(self, in_size, out_size, kernel = 3, dilation = 1):
        super().__init__()

        self.norm = nn.BatchNorm2d(in_size)
        self.conv1 = nn.ConvTranspose2d(in_size, out_size, kernel, dilation = dilation)


    def forward(self, inputs):
        return F.relu(self.conv1(self.norm(inputs)))

class Encode(nn.Module):

    def __init__(self, in_size, out_size):
        super().__init__()
        self.conv = Conv(in_size, out_size)

    def forward(self, inputs):
        output = self.conv(inputs)
        return F.max_pool2d(output, 2, 2, ceil_mode = True), output


def sub(xs, ys):
     return tuple (x - y for x, y in zip(xs, ys))


def match_size_2d(t, sized):
    assert t.dim() == 4 and sized.dim() == 4
    dh = sized.size(2) - t.size(2)
    dw = sized.size(3) - t.size(3)

    pad = (dw // 2, dw - dw // 2, dh // 2, dh - dh // 2)
    return F.pad(t, pad)



class Decode(nn.Module):

    def __init__(self, in_size, out_size):
        super().__init__()

        self.conv = ConvT(in_size + out_size, out_size)
        self.up = nn.ConvTranspose2d(in_size, out_size, 2, 2)

    def forward(self, inputs, skip):
        upscaled = self.up(inputs)
        padded = match_size_2d(upscaled, skip)

        return self.conv(torch.cat([padded, skip], 1))


segmenter = pyramid.segmenter(Encode, Decode)

import torch
import torch.nn as nn
import torch.nn.functional as F

import tools.pyramid as pyramid

class Conv2(nn.Module):
    def __init__(self, in_size, out_size, kernel = 3):
        super().__init__()

        self.conv1 = Conv(in_size, out_size, kernel)
        self.conv2 = Conv(out_size, out_size, kernel)

    def forward(self, inputs):
        return self.conv2(self.conv1(inputs))

class Conv(nn.Module):

    def __init__(self, in_size, out_size, kernel = 3):
        super().__init__()

        self.norm = nn.BatchNorm2d(in_size)
        self.conv1 = nn.Conv2d(in_size, out_size, 3, 1, 1)

    def forward(self, inputs):
        return F.relu(self.conv1(self.norm(inputs)))

Convs = Conv2

class Encode(nn.Module):

    def __init__(self, in_size, out_size):
        super().__init__()
        self.conv = Convs(in_size, out_size)

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

        self.conv = Convs(in_size + out_size, out_size)
        self.up = nn.ConvTranspose2d(in_size, out_size, 2, 2)

    def forward(self, inputs, skip):
        upscaled = self.up(inputs)
        padded = match_size_2d(upscaled, skip)

        return self.conv(torch.cat([padded, skip], 1))


segmenter = pyramid.segmenter(Encode, Decode)

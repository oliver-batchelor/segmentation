import torch
import torch.nn as nn
import torch.nn.functional as F

import tools.pyramid as pyramid

class Convs(nn.Module):

    def __init__(self, in_size, out_size):
        super().__init__()

        self.norm = nn.BatchNorm2d(in_size)
        self.conv1 = nn.Conv2d(in_size, out_size, 3, 1, 1)

    def forward(self, inputs):

        return F.relu(self.conv1(self.norm(inputs)))

class Encode(nn.Module):

    def __init__(self, in_size, out_size):
        super().__init__()
        self.conv = Convs(in_size, out_size)

    def forward(self, inputs):
        return F.max_pool2d(self.conv(inputs), 2, 2, return_indices = True)


def sub(xs, ys):
     return tuple (x - y for x, y in zip(xs, ys))


def match_size(t, sized):
    assert(t.dim() == sized.dim())
    return F.pad(t, sub(sized.size(), t.size()))



class Decode(nn.Module):

    def __init__(self, in_size, out_size):
        super().__init__()

        self.conv = Convs(in_size, out_size)

    def forward(self, inputs, indices):

        upscaled = F.max_unpool2d(inputs, indices, 2, 2)
        return self.conv(upscaled)

def segmenter(**kwargs):
    return pyramid.segmenter(Encode, Decode, **kwargs)

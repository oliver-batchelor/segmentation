import torch
import torch.nn as nn
import torch.nn.functional as F

import tools.model.pyramid as pyramid
from tools.model import match_size_2d


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



class Decode(nn.Module):

    def __init__(self, in_size, out_size):
        super().__init__()

        self.conv = Convs(in_size, out_size)

    def forward(self, inputs, indices):

        padded = match_size_2d(inputs, indices)


        upscaled = F.max_unpool2d(padded, indices, 2, 2)
        return self.conv(upscaled)

segmenter = pyramid.segmenter(Encode, Decode)

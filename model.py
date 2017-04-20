import torch
import torch.nn as nn
import torch.nn.functional as F



class Convs(nn.Module):

    def __init__(self, in_size, out_size):
        super().__init__()

        self.conv1 = nn.Conv2d(in_size, out_size, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_size, out_size, 3, 1, 1)

    def forward(self, inputs):
        outputs = F.relu(self.conv1(inputs))
        return F.relu(self.conv2(outputs))


class Encode(nn.Module):

    def __init__(self, in_size, out_size):
        super().__init__()
        self.conv = Convs(in_size, out_size)

    def forward(self, inputs):
        return F.max_pool2d(self.conv(inputs), 2, 2)


def sub(xs, ys):
     return tuple (x - y for x, y in zip(xs, ys))


def match_size(t, sized):
    assert(t.dim() == sized.dim())
    return F.pad(t, sub(sized.size(), t.size()))



class Decode(nn.Module):

    def __init__(self, in_size, out_size):
        super().__init__()

        self.conv = Convs(in_size, out_size)
        self.up = nn.ConvTranspose2d(in_size, out_size, 2, 2)

    def forward(self, skip, inputs):
        upscaled = self.up(inputs)
        padded = match_size(upscaled, skip)

        return self.conv(torch.cat([padded, skip], 1))


class Segmenter(nn.Module):

    def __init__(self, num_classes, input_channels = 3, num_channels = 1):
        super().__init__()

        self.encode1 = Encode(input_channels, 32*num_channels)
        self.encode2 = Encode(32*num_channels, 64*num_channels)
        self.encode3 = Encode(64*num_channels, 128*num_channels)
        self.center = Encode(128*num_channels, 256*num_channels)
        self.decode3 = Decode(256*num_channels, 128*num_channels)
        self.decode2 = Decode(128*num_channels, 64*num_channels)
        self.decode1 = Decode(64*num_channels, 32*num_channels)
        self.final = nn.Conv2d(32*num_channels, num_classes, 1)

    def forward(self, inputs):
        encode1 = self.encode1(inputs)
        encode2 = self.encode2(encode1)
        encode3 = self.encode3(encode2)
        centre = self.center(encode3)


        decode3 = self.decode3(encode3, centre)
        decode2 = self.decode2(encode2, decode3)
        decode1 = self.decode1(encode1, decode2)
        return F.log_softmax(self.final(decode1))

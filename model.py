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
        outputs = F.relu(self.conv2(outputs))

        return outputs


class Encode(nn.Module):

    def __init__(self, in_size, out_size):
        super().__init__()

        self.conv = Convs(in_size, out_size)
        self.down = nn.MaxPool2d(2, 1)

    def forward(self, inputs):
        outputs = self.conv(inputs)
        outputs = self.down(outputs)

        return outputs


class Decode(nn.Module):

    def __init__(self, in_size, out_size):
        super().__init__()

        self.conv = Convs(in_size, out_size)
        self.up = nn.ConvTranspose2d(in_size, out_size, 2)

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)

        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2*[offset // 2, offset // 2 + 1]

        outputs1 = F.pad(inputs1, padding)

        return self.conv(torch.cat([outputs1, outputs2], 1))


class Segmenter(nn.Module):

    def __init__(self, num_classes, input_channels = 3, num_channels = 1):
        super().__init__()

        self.encode1 = Encode(input_channels, 32*num_channels)
        self.encode2 = Encode(32*num_channels, 64*num_channels)
        self.encode3 = Encode(64*num_channels, 128*num_channels)
        self.center = Convs(128*num_channels, 256*num_channels)
        self.decode3 = Decode(256*num_channels, 128*num_channels)
        self.decode2 = Decode(128*num_channels, 64*num_channels)
        self.decode1 = Decode(64*num_channels, 32*num_channels)
        self.final = nn.Conv2d(32*num_channels, num_classes, 1)

    def forward(self, inputs):
        encode1 = self.encode1(inputs)
        encode2 = self.encode2(encode1)
        encode3 = self.encode3(encode2)
        center = self.center(encode3)
        decode3 = self.decode3(encode3, center)
        decode2 = self.decode2(encode2, decode3)
        decode1 = self.decode1(encode1, decode2)

        return self.final(decode1)

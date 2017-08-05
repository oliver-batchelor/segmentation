import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from functools import partial
from tools.model import match_size_2d,  centre_crop



def identity(x, **kwargs):
    return x



class Cascade(nn.Sequential):
    def __init__(self, *args):
        super(Cascade, self).__init__(*args)

    def forward(self, input):
        outputs = [input]

        for module in self._modules.values():
            input = module(input)
            outputs.append(input)

        return outputs

class DeCascade(nn.Module):
    def __init__(self, bottom, *decoders):
        super(DeCascade, self).__init__()

        self.bottom = bottom
        self.decoders = nn.Sequential(*decoders)

    def forward(self, inputs):
        assert len(inputs) == len(self.decoders) + 1
        input, *skips = list(reversed(inputs))

        input = self.bottom(input)

        for module, skip in zip(self.decoders._modules.values(), skips):
            input = module(input, skip)
        return input




class Residual(nn.Sequential):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, input):
        output = self.module(input)

        d = output.size(1) - input.size(1)
        if d > 0:   # zero padded skip connection
            input = output.narrow(1, 0, input.size(1)) + input
            output = output.narrow(1, input.size(1), d)
            return torch.cat([input, output], 1)
        elif d < 0: # truncated skip conection
            return input.narrow(1, 0, output.size(1)) + output

        return output + input



class Lift(nn.Module):
    def __init__(self, f, **kwargs):
        super().__init__()

        self.kwargs = kwargs
        self.f = f

    def forward(self, input):
        return self.f(input, **self.kwargs)


class Conv(nn.Module):
    def __init__(self, in_size, out_size, kernel=3, stride=1, padding=None, bias=False, activation=nn.ReLU(inplace=True), groups=1):
        super().__init__()

        padding = kernel//2 if padding is None else padding

        self.norm = nn.BatchNorm2d(in_size)
        self.conv = nn.Conv2d(in_size, out_size, kernel, stride=stride, padding=padding, bias=bias, groups=1)
        self.activation = activation

    def forward(self, inputs):
        return self.conv(self.activation(self.norm(inputs)))


def basic_block(in_size, out_size):
    unit = nn.Sequential(Conv(in_size, out_size, activation=identity), Conv(out_size, out_size))
    return Residual(unit)


class Decode(nn.Module):
    def __init__(self, in_size, skip_size, out_size, module, scale_factor=2):
        super().__init__()
        self.reduce = Conv(skip_size + in_size, out_size, 1)
        self.scale_factor = scale_factor
        self.module = module

    def forward(self, inputs, skip):
        upscaled = F.upsample(inputs, scale_factor=self.scale_factor)
        upscaled = match_size_2d(upscaled, skip)
        inputs = torch.cat([upscaled, skip], 1)
        inputs = self.reduce(inputs)

        return self.module(inputs)


class Decoder(nn.Module):
    def __init__(self, encoder_sizes, base_features, num_blocks=1, block=basic_block, scale_factor=2):
        super().__init__()

        sizes = list(reversed(encoder_sizes))
        def features(i):
            return base_features * 2 ** (len(encoder_sizes) - i - 1)

        def layer(i):
            n = features(i)
            processing = nn.Sequential(*[block(n, n) for x in range(0, num_blocks)])
            return Decode(features(i - 1), sizes[i], n, processing, scale_factor=scale_factor)

        layers = [layer(i) for i in range(1, len(sizes))]
        self.decoder = DeCascade(block(sizes[0], features(0)), *layers)

    def forward(self, inputs):
        return self.decoder(inputs)

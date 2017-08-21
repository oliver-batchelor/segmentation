import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

from functools import partial
from tools.model import match_size_2d,  centre_crop



def identity(x, **kwargs):
    return x

def reverse(xs):
    return list(reversed(xs))



class Lift(nn.Module):
    def __init__(self, f, **kwargs):
        super().__init__()

        self.kwargs = kwargs
        self.f = f

    def forward(self, input):
        return self.f(input, **self.kwargs)

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


def cascade(modules, input):
    outputs = []

    for module in modules:
        input = module(input)
        outputs.append(input)

    return outputs

class Cascade(nn.Sequential):
    def __init__(self, *args):
        super(Cascade, self).__init__(*args)

    def forward(self, input):
        return cascade(self._modules.values(), input)

class MultiCascade(nn.Module):
    def __init__(self, cascade, levels=None, depth=None):
        super(MultiCascade, self).__init__()

        self.depth = depth or len(modules)
        self.levels = levels or len(modules)

        self.cascade = cascade

    def forward(self, input):
        modules = list(self.cascade._modules.values())
        outputs = [[] for x in range(0, self.depth)]

        for i in range(0, self.levels):
            output = cascade(modules[:self.depth - i], input)
            input  = F.avg_pool2d(input, 2, 2, ceil_mode=True)

            for j in range(0, len(output)):
                print(i + j, outputs[j].size())
                outputs[j + i] += [output[j]]

        return [torch.cat(output, 1) for output in outputs if len(output) > 0]



class DeCascade(nn.Module):
    def __init__(self, bottom, *decoders):
        super(DeCascade, self).__init__()

        self.bottom = bottom
        self.decoders = nn.Sequential(*decoders)

    def forward(self, inputs):

        assert len(inputs) == len(self.decoders) + 1
        input, *skips = reverse(inputs)

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





class Conv(nn.Module):
    def __init__(self, in_size, out_size, kernel=3, stride=1, padding=None, bias=False, activation=nn.ReLU(inplace=True), groups=1):
        super().__init__()

        padding = kernel//2 if padding is None else padding

        self.norm = nn.BatchNorm2d(in_size)
        self.conv = nn.Conv2d(in_size, out_size, kernel, stride=stride, padding=padding, bias=bias, groups=1)
        self.activation = activation

    def forward(self, inputs):
        return self.conv(self.activation(self.norm(inputs)))


def dropout(p=0.0):
    return nn.Dropout2d(p=p) if p > 0 else Lift(identity)

def basic_block(p=0):
    def f(in_size, out_size):
        unit = nn.Sequential(Conv(in_size, out_size, activation=identity), dropout(p=p), Conv(out_size, out_size), nn.BatchNorm2d(out_size))
        return Residual(unit)
    return f


def bottleneck_block(p=0):
    def f(in_size, out_size):
        unit = nn.Sequential(Conv(in_size, out_size//4, 1, activation=identity), dropout(p=p), Conv(out_size//4, out_size, 3), nn.BatchNorm2d(out_size))
        return Residual(unit)
    return f

def reducer(in_size, out_size, depth=3):
    def interp(i):
        t = i / depth
        d = out_size + int((1 - t) * (in_size - out_size))
        return d

    return nn.Sequential(*[Conv(interp(i), interp(i + 1), 1) for i in range(0, depth)])


def unbalanced_add(x, y):
    if x.size(1) > y.size(1):
        x = x.narrow(0, 1, y.size(1))
    elif y.size(1) < x.size(1):
        y = y.narrow(0, 1, x.size(1))

    return x + y



class Decode(nn.Module):
    def __init__(self, in_size, skip_size, out_size, module, scale_factor=2):
        super().__init__()
        self.reduce = reducer(skip_size + in_size, out_size)
        self.scale_factor = scale_factor
        self.module = module

    def forward(self, inputs, skip):
        upscaled = F.upsample(inputs, scale_factor=self.scale_factor)
        upscaled = match_size_2d(upscaled, skip)

        inputs = torch.cat([upscaled, skip], 1)
        inputs = self.reduce(inputs)

        return self.module(inputs)



def make_blocks(block, features, size):
    return [block(features, features) for x in range(0, size)]

def decoder(base_features, inc_features, encoder_sizes, block_sizes, decode=Decode, block=basic_block(p=0), scale_factor=2):
    encoder_sizes, block_sizes = reverse(encoder_sizes), reverse(block_sizes)
    depth = len(block_sizes)
    assert len(encoder_sizes) == depth

    def features(i):
        return base_features + inc_features * (depth - i - 1)

    def layer(i):
        blocks = nn.Sequential(*make_blocks(block, features(i), block_sizes[i]))
        return decode(features(i - 1), encoder_sizes[i], features(i), blocks, scale_factor=scale_factor)

    layers = [layer(i) for i in range(1, depth)]
    bottom = nn.Sequential(Conv(encoder_sizes[0], features(0), 1), *make_blocks(block, features(0), block_sizes[0]))

    return DeCascade(bottom, *layers)

def encoder(base_features, inc_features, block_sizes, block=basic_block(p=0), scale_factor=2):
    depth = len(block_sizes)

    def features(i):
        return base_features + inc_features * i

    def layer(i):
        blocks = make_blocks(block, features(i), block_sizes[i] - 1)
        initial = block(features(i - 1), features(i))
        return nn.Sequential(nn.AvgPool2d(scale_factor, scale_factor), initial, *blocks)

    top = nn.Sequential(*make_blocks(block, features(0), block_sizes[0]))

    sizes = [features(i) for i in range(0, depth)]
    return Cascade(top, *[layer(i) for i in range(1, depth)]), sizes


def init_weights(module):
    def f(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
            init.kaiming_normal(m.weight)
    module.apply(f)

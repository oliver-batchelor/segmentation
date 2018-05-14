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






def map_modules(m, type, f):
    if isinstance(m, type):
        return f(m)

    for k, v in m._modules.items():
        m._modules[k] = map_modules(m._modules[k], type, f)

    return m

def replace_batchnorms(m, num_groups):
    def convert(b):
        g = nn.GroupNorm(num_groups, b.num_features)
        g.weight = b.weight
        g.bias = b.bias

        return g

    return map_modules(m, nn.BatchNorm2d, convert)


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


# class StepDown(nn.Module):
#     def __init__(self, in_size, out_size, steps=1, kernel=3, bias=False, activation=nn.ReLU(inplace=True)):
#
#         sizes = []
#
#         self.module = nn.Sequential()



class UpCascade(nn.Module):
    def __init__(self, *decoders):
        super(UpCascade, self).__init__()

        self.decoders = nn.ModuleList(*decoders)

    def forward(self, inputs):

        assert len(inputs) == len(self.decoders)

        input = None
        outputs = []

        for module, skip in zip(reverse(self.decoders._modules.values()), reverse(inputs)):
            input = module(input, skip)
            outputs.append(input)

        return reverse(outputs)


class Parallel(nn.Module):
    def __init__(self, *modules):
        super(Parallel, self).__init__()
        self.mods = nn.ModuleList(*modules)

    def forward(self, inputs):
        assert len(inputs) == len(self.mods)
        return [m(i) for m, i in zip(self.mods, inputs)]


class Shared(nn.Module):
    def __init__(self, module):
        super(Shared, self).__init__()
        self.module = module

    def forward(self, inputs):
        return [self.module(input) for  input in inputs]


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

def basic_block(in_size, out_size):
    unit = nn.Sequential(Conv(in_size, out_size, activation=identity), Conv(out_size, out_size), nn.BatchNorm2d(out_size))
    return Residual(unit)

def bottleneck_block(in_size, out_size):
    unit = nn.Sequential(Conv(in_size, out_size//4, 1, activation=identity), Conv(out_size//4, out_size, 3), nn.BatchNorm2d(out_size))
    return Residual(unit)

def reduce_features(in_size, out_size, steps=2, kernel=1):
    def interp(i):
        t = i / steps
        d = out_size + int((1 - t) * (in_size - out_size))
        return d

    m = nn.Sequential(*[Conv(interp(i), interp(i + 1), kernel) for i in range(0, steps)])
    return m


def unbalanced_add(x, y):
    if x.size(1) > y.size(1):
        x = x.narrow(0, 1, y.size(1))
    elif y.size(1) < x.size(1):
        y = y.narrow(0, 1, x.size(1))

    return x + y


class Upscale(nn.Module):
    def __init__(self, features, scale_factor=2):
        super().__init__()
        self.conv = Conv(features, features*4)
        self.scale_factor = scale_factor

    def forward(self, inputs):
        return F.pixel_shuffle(self.conv(inputs), self.scale_factor)


class DecodeAdd(nn.Module):
    def __init__(self, features, module=None, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor
        self.module = module or identity
        #self.upscale = nn.Upsample(scale_factor=scale_factor, mode='nearest')
        self.upscale = Upscale(features, scale_factor=scale_factor)

    def forward(self, inputs, skip):
        if not (inputs is None):
            upscaled = self.upscale(inputs)
            upscaled = match_size_2d(upscaled, skip)
            return self.module(skip + upscaled)

        return self.module(skip)

class Decode(nn.Module):
    def __init__(self, features, module=None, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor
        self.reduce = Conv(features * 2, features)
        self.module = module or identity
        #self.upscale = nn.Upsample(scale_factor=scale_factor, mode='nearest')
        self.upscale = Upscale(features, scale_factor=scale_factor)


    def forward(self, inputs, skip):
        if not (inputs is None):
            #upscaled = F.upsample(inputs, scale_factor=self.scale_factor)
            upscaled = self.upscale(inputs)
            upscaled = match_size_2d(upscaled, skip)

            return self.module(self.reduce(torch.cat([upscaled, skip], 1)))

        return self.module(skip)



def init_weights(module):
    def f(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
            init.kaiming_normal(m.weight)
    module.apply(f)

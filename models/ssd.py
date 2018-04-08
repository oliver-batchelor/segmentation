import sys
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import itertools
import torchvision.models as m

import models.pretrained as pretrained
from detection import anchors

from models.common import Conv, Cascade, Residual
import tools.model.io as io

from tools import Struct


class Encoder:
    def __init__(self, start_layer, box_sizes, nms_threshold=0.5, class_threshold=0.05):
        self.anchor_cache = {}

        self.box_sizes = box_sizes
        self.start_layer = start_layer

        self.nms_threshold = nms_threshold
        self.class_threshold = class_threshold

    def anchors(self, input_size):
        def layer_size(i):
            scale = 2 ** i
            return (max(1, math.floor(input_size[0] / scale)), max(1, math.floor(input_size[1] / scale)))

        if not (input_size in self.anchor_cache):
            layer_dims = [layer_size(self.start_layer + i) for i in range(0, len(self.box_sizes))]
            self.anchor_cache[input_size] = anchors.make_anchors(self.box_sizes, layer_dims, input_size)

        return self.anchor_cache[input_size]

    def encode(self, inputs, boxes, labels):

        if torch.is_tensor(inputs):
            assert(inputs.dim() == 3)
            inputs = image.size(2), image.size(1)

        return anchors.encode(boxes, labels, self.anchors(inputs))


    def decode(self, inputs, loc_pred, class_pred):
        assert loc_pred.dim() == 2 and class_pred.dim() == 2
        if torch.is_tensor(inputs):
            assert(inputs.dim() == 3)
            inputs = inputs.size(2), inputs.size(1)

        assert len(inputs) == 2

        return anchors.decode(loc_pred, class_pred, self.anchors(inputs).type_as(loc_pred),
               nms_threshold=self.nms_threshold, class_threshold=self.class_threshold)


    def decode_batch(self, inputs, loc_pred, class_pred):
        assert loc_pred.dim() == 3 and class_pred.dim() == 3

        if torch.is_tensor(inputs):
            assert(inputs.dim() == 4)
            inputs = inputs.size(3), inputs.size(2)

        assert len(inputs) == 2
        return [self.decode(inputs, l, c) for l, c in zip(loc_pred, class_pred)]


class SSD(nn.Module):

    def __init__(self, layers, box_sizes, num_classes=2):
        super().__init__()
        self.encoder = pretrained.make_cascade(layers)
        self.box_sizes = box_sizes

        features = pretrained.encoder_sizes(self.encoder)

        print( len(features), len(box_sizes))
        assert len(layers) == len(box_sizes), "layers and box sizes differ in length"

        classifiers = [Conv(size, len(boxes) * (num_classes + 5)) for size, boxes in zip (features, self.box_sizes)]

        self.classifiers = nn.ModuleList(classifiers)
        self.num_classes = num_classes



        # self.anchor_cache = []


    def forward(self, input):
        skips = self.encoder(input)

        # print(skips)

        def classify(layer, skip):
            out = layer(skip).permute(0, 2, 3, 1).contiguous()
            return out.view(out.size(0), -1, self.num_classes + 5)

        out = [classify(*x) for x in zip (self.classifiers, skips)]
        out = torch.cat(out, 1)

        # centres, sizes, conf = out.narrow(2, 0, 2), out.narrow(2, 2, 2), out.narrow(2, 4, self.num_classes)
        # sizes = sizes.abs()
        # loc = torch.cat([centres, sizes], 2)
        loc, conf = out.narrow(2, 0, 4), out.narrow(2, 4, self.num_classes + 1)
        return (loc, conf)


        #return self.decoder(skips)


    def parameter_groups(self, args):
        return [
                {'params': self.encoder.parameters(), 'lr': self.modifier * args.lr, 'modifier': self.modifier},
                {'params': self.decoder.parameters()}
            ]


parameters = Struct(
        base_name       = ("resnet18", "name of pretrained resnet to use"),
        lr_modifier     = (0.1, "learning rate modifier for pretrained part (fine tuning)"),

        first   = (3, "first layer of anchor boxes, anchor size = 2^(n + 2)"),
        last    = (4, "last layer of anchor boxes")
    )

def extra_layer(features):
    convs = nn.Sequential (Conv(features, features), Conv(features, features))
    return nn.Sequential(Residual(convs), Conv(features, features, stride=2))

def split_at(xs, n):
    return xs[:n], xs[n:]


def anchor_sizes(start, end):

    aspects = [1/2, 1, 2]
    scales = [1, pow(2, 1/3), pow(2, 2/3)]

    return [anchors.anchor_sizes(2 ** (i + 2), aspects, scales) for i in range(start, end + 1)]


def extend_layers(layers, start, end):
    sizes = pretrained.layer_sizes(layers)

    num_extra = max(0, end + 1 - len(layers))
    extra_layers = [extra_layer(sizes[-1]) for i in range(0, num_extra)]

    initial, rest =  layers[:start + 1], layers[start + 1:end + 1:]
    return [nn.Sequential(*initial), *rest, *extra_layers]


def create_ssd(args, num_classes=2, input_channels=3):
    assert input_channels == 3

    layers = extend_layers(pretrained.get_layers(args.base_name), args.first, args.last)
    box_sizes = anchor_sizes(args.first, args.last)

    return SSD(layers, box_sizes, num_classes=num_classes), Encoder(args.first, box_sizes, nms_threshold=0.3)


models = {
    'ssd' : Struct(create=create_ssd, parameters=parameters)
  }

if __name__ == '__main__':

    _, *cmd_args = sys.argv
    args = io.parse_params(models, cmd_args)

    model = create_ssd(args, 2, 3)

    x = Variable(torch.FloatTensor(4, 3, 600, 600))
    out = model.cuda()(x.cuda())

    [print(y.size()) for y in out]

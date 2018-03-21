import sys
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import itertools
import torchvision.models as m

import models.pretrained as pretrained

from models.common import Conv, Cascade, Residual
import tools.model.io as io

from tools import Struct

    


class SSD(nn.Module):

    def __init__(self, args, layers, prior_sizes, num_classes=2):
        super().__init__()
        self.encoder = pretrained.make_cascade(layers)
        self.prior_sizes = prior_sizes
        
        sizes = pretrained.encoder_sizes(self.encoder)
        assert len(sizes) == len(prior_sizes), "encoder layers and layer_boxes differ in length"
        classifiers = [Conv(size, len(boxes) * (num_classes + 4)) for size, boxes in zip (sizes, self.prior_sizes)]
        
        self.classifiers = nn.ModuleList(classifiers)
        self.num_classes = num_classes    
    
        self.prior_boxes = {}
        
        # self.prior_cache = []
    

    def forward(self, input):
        skips = self.encoder(input)
        
        # print(skips)
        
        def classify(layer, skip):
            out = layer(skip).permute(0, 2, 3, 1).contiguous()
            return out.view(out.size(0), -1, self.num_classes + 4)
        
        out = [classify(*x) for x in zip (self.classifiers, skips)]
        out = torch.cat(out, 1)
                
        # centres, sizes, conf = out.narrow(2, 0, 2), out.narrow(2, 2, 2), out.narrow(2, 4, self.num_classes)
        # sizes = sizes.abs()        
        # loc = torch.cat([centres, sizes], 2)
        loc, conf = out.narrow(2, 0, 4), out.narrow(2, 4, self.num_classes)
        
        image_size = lambda s: (s.size(3), s.size(2))
        layer_dims = [image_size(s) for s in skips]
        
        input_size = image_size(input)
        
        priors = priors[input_size] if input_size in self.prior_boxes else make_prior_boxes(self.prior_sizes, layer_dims)
        priors = priors.type_as(input.data)
        self.prior_boxes[input_size] = priors    
        
        return (loc, conf, Variable(priors, requires_grad=False))
        
    
        #return self.decoder(skips)


    def parameter_groups(self, args):
        return [
                {'params': self.encoder.parameters(), 'lr': self.modifier * args.lr, 'modifier': self.modifier},
                {'params': self.decoder.parameters()}
            ]


parameters = Struct(
        base_name       = ("resnet18", "name of pretrained resnet to use"),
        lr_modifier     = (0.1, "learning rate modifier for pretrained part (fine tuning)"),
        
        initial_layers  = (4, "layers not used for classification"),
        extra_layers    = (3, "extra layers"),
        
        min_size = (0.1, "smallest prior box size (as proportion of training image size)"),
        max_size = (0.9, "largest prior box size (as proportion of training image size)"),
    )

def extra_layer(features):
    convs = nn.Sequential (Conv(features, features), Conv(features, features))
    return nn.Sequential(Residual(convs), Conv(features, features, stride=2))    
    
def split_at(xs, n):
    return xs[:n], xs[n:]    


# def box_sizes(aspects):
# 
#     f = lambda ar: (s_k/math.sqrt(ar), s_k * math.sqrt(ar))
# 
# 
#     for ar in aspects:
    

def make_boxes(sizes, width, height):
    
    n = len(sizes)
    xs = torch.arange(0, width).add_(0.5).view(1, width, 1, 1).expand(height, width, n, 1)
    ys = torch.arange(0, height).add_(0.5).view(height, 1, 1, 1).expand(height, width, n, 1) 
    
    box_sizes = torch.Tensor(sizes).view(1, 1, n, 2).expand(height, width, n, 2)    
    return torch.cat([xs, ys, box_sizes], 3).view(-1, 4)

    

def make_prior_boxes(prior_sizes, layer_dims):
    
    boxes = [make_boxes(boxes, *dim) for boxes, dim in zip(prior_sizes, layer_dims)]
    return torch.cat(boxes, 0).clamp_(0, 1)


        

def make_prior_sizes(min_size, max_size, num_boxes):
    size = lambda k: min_size + (max_size - min_size) * k / len(num_boxes)
        
    def layer_boxes(k):
        s = size(k)
        aspect = lambda ar: (s * math.sqrt(ar), s / math.sqrt(ar)) 
        
        s1 = math.sqrt(s * size(k + 1))
        return [ (s1, s1), aspect(1)
                , aspect(2), aspect(1/2)
                , aspect(3), aspect(1/3) ]
        
    return [layer_boxes(k)[:n] for k, n in enumerate(num_boxes)]

    
def create_ssd(args, num_classes=2, input_channels=3):
    assert input_channels == 3
    pretrained_layers = pretrained.get_layers(args.base_name)
    
    initial, rest = split_at(pretrained_layers, args.initial_layers)
    
    sizes = pretrained.layer_sizes(pretrained_layers)
    extra_layers = [extra_layer(sizes[-1]) for i in range(0, args.extra_layers)]
    
    layers = [nn.Sequential(*initial), *rest, *extra_layers]

    num_boxes = [(6 if i > 0 and i < 4 else 4) for i in range(0, len(layers))]
    prior_sizes =  make_prior_sizes(args.min_size, args.max_size, num_boxes)        

    return SSD(args, layers, prior_sizes, num_classes=num_classes)


models = {
    'ssd' : Struct(create=create_ssd, parameters=parameters)
  }

if __name__ == '__main__':

    _, *cmd_args = sys.argv
    args = io.parse_params(models, cmd_args)

    model = create_ssd(args, 2, 3)

    x = Variable(torch.FloatTensor(4, 3, 500, 500))
    out = model.cuda()(x.cuda())
    
    [print(y.size()) for y in out]

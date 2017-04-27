from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
import argparse

# from torch.autograd import Variable

import os
import dataset

from model import Segmenter
from tools import model_io, loaders, transforms, index_map
import tools

from torch.autograd import Variable


parser = argparse.ArgumentParser(description='Tree segmentation test')
parser.add_argument('image', help='image to load')

parser.add_argument('--save', help='save result to file with name')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
args = parser.parse_args()

model = Segmenter(2)

epoch, state = model_io.load('models')
model.load_state_dict(state)

args.cuda = not args.no_cuda and torch.cuda.is_available()
model = Segmenter(2)
if args.cuda:
    model.cuda()


try:
    image = loaders.load_rgb(args.image)
except:
    print("could not load image ", args.image)
    exit(1)

input_size = (256, 256)
image = image.crop((0, 0, 600, 600)).resize(input_size, Image.BICUBIC)


if image:
    image_tensor = transforms.to_tensor(image)

    data = image_tensor.view(1, *image_tensor.size())
    if args.cuda:
        data = data.cuda()

    model.eval()
    output = model(Variable(data)).data
    _, inds = output.max(1)

    inds = inds.byte()

    if(args.save):
        result = tools.tensor.to_image(inds)
        result = image.resize((data.size(3), data.size(2)), Image.NEAREST)

        result.save(args.save)
    else:
        color_map = index_map.make_color_map(255)
        overlay = index_map.overlay_labels(image_tensor, inds, color_map)
        overlay.show()



#if(args.show):

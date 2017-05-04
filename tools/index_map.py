
import torch
import colorsys
import random
import math

from tools import tensor
from PIL import Image


def make_color_map(n):
    colors = torch.ByteTensor(n, 3).fill_(0)

    for i in range(1, n): 
        h = i / n
        s = (2 + (i // 2 % 2)) / 3
        v = (2 + (i % 2)) / 3

        rgb = torch.FloatTensor (colorsys.hsv_to_rgb(h, s, v))
        colors[i] = (rgb * 255).byte()

    d = int(math.sqrt(n))
    p = torch.arange(0, n).long().view(d, -1).t().contiguous().view(n)
    return colors.index(p)

default_map = make_color_map(255)

def colorize(image, color_map):
    assert(image.dim() == 3 and image.size(2) == 1)

    flat_indices = image.view(image.nelement()).long()
    rgb = color_map.index(flat_indices)

    return rgb.view(image.size(0), image.size(1), 3)

def colorize_t(image, color_map):
    assert(image.dim() == 3 and image.size(0) == 1)
    return colorize(image.permute(1, 2, 0), color_map).permute(2, 0, 1)

def colorizer(n = 255):

    color_map = make_color_map(n)
    return lambda image: colorize(image, color_map)

def colorizer_t(n = 255):

    color_map = make_color_map(n)
    return lambda image: colorize_t(image, color_map)




def overlay_labels(image, labels, color_map = default_map):

    assert(image.dim() == 3 and image.size(0) == 3)

    if(labels.dim() == 2):
        labels = labels.view(1, *labels.size())

    assert(labels.dim() == 3 and labels.size(0) == 1)

    dim = (image.size(2), image.size(1))

    labels_color = tensor.to_image_t(colorize_t(labels, color_map)).resize(dim, Image.NEAREST)
    labels = labels.clamp_(0, 1).mul(255)

    mask = tensor.to_image(labels.squeeze(0).byte(), 'L').resize(dim, Image.NEAREST)
    image = tensor.to_image_t(image.mul(255).byte())

    return Image.composite(labels_color, image, mask)



def overlay_batches(images, target, cols = 6, color_map = default_map):

    images = tensor.tile_batch(images, cols)
    target = target.view(target.size(0), 1, target.size(1), target.size(2))

    target = tensor.tile_batch(target, cols)

    return overlay_labels(images, target, color_map)

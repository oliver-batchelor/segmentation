
import torch
import colorsys
import random
import math

from tools import tensor
import tools.cv as cv


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




def overlay_labels(image, labels, color_map = default_map):
    assert(image.dim() == 3 and image.size(2) == 3)

    if(labels.dim() == 2):
        labels = labels.view(*labels.size(), 1)

    assert(labels.dim() == 3 and labels.size(2) == 1)
    dim = (image.size(1), image.size(0))
    labels = cv.resize(labels, dim, interpolation = cv.INTER_NEAREST)

    labels_color = colorize(labels, color_map).float()
    mask = 0.5 * labels.clamp_(0, 1).expand_as(labels_color).float()

    return (image.float() * (1 - mask) + labels_color * mask).type_as(image)


def overlay_batches(images, target, cols = 6, color_map = default_map):
    images = tensor.tile_batch(images, cols)

    target = target.view(*target.size(), 1)
    target = tensor.tile_batch(target, cols)

    return overlay_labels(images, target, color_map)

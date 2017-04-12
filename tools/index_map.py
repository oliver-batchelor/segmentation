
import torch
import colorsys
import random
import math



def make_color_map(n):
    colors = torch.ByteTensor(n, 3).fill_(0)

    for i in range(1, n):
        h = i / n
        s = (2 + (i // 2 % 2)) / 3
        v = (2 + (i % 2)) / 3

        rgb = torch.FloatTensor (colorsys.hsv_to_rgb(h, s, v))
        colors[i] = (rgb * 255).byte()

    d = int(math.sqrt(n))
    p = torch.range(0, n - 1).long().view(d, -1).t().contiguous().view(n)
    return colors.index(p)



def colorize(image, color_map):
    assert(image.size(2) == 1)

    flat_indices = image.view(image.nelement()).long()
    rgb = color_map.index(flat_indices)

    return rgb.view(image.size(0), image.size(1), 3)

def colorizer(n = 255):

    color_map = make_color_map(n)
    return lambda image: colorize(image, color_map)

def colorizer_t(n = 255):

    color_map = make_color_map(n)
    def f(image):
        assert(image.dim() == 3 and image.size(0) == 1)
        return colorize(image.permute(1, 2, 0), color_map).permute(2, 0, 1)

    return f


def display_labels(image, labels):
    print(image, labels)

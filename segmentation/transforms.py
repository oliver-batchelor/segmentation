
import torch
import random
import math

from tools import tensor, Struct
from tools.image import cv, transforms



def adjust_channel(image, gamma_range = 0.1):
    gamma = random.uniform(1-gamma_range, 1+gamma_range)
    return cv.adjust_gamma(image, gamma)


def adjust_colors(gamma_range = 0.1):
    def f(image):
        for d in range(0, 2):
            image.select(2, d).copy_(adjust_channel(image.select(2, d), gamma_range))

        return image
    return modify(f, 'image')


def scale_weights(scale):
    def f(weights):
        return weights.pow(scale)
    return modify(f, 'weight')



def compose(fs):
    def composed(data):
        for f in fs:
            data = f(data)
        return data
    return composed


def identity(image):
    return image

default_statistics = Struct(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])

def normalize(image, mean=default_statistics.mean, std=default_statistics.mean):
    image = image.float().div_(255)
    for i in range(0, 2):
        image.select(2, i).sub_(mean[i]).div_(std[i])
    return image.permute(0, 3, 1, 2)


def normalize_target(target, num_classes, weights=None):
    mask = target < num_classes
    weights = (weights or 1) * mask.float()

    return target.long() * mask.long(), weights


def warp_affine(data, t, dest_size, border_fill=None):
    image, target, weight = data['image'], data['target'], data['weight']

    image = cv.warpAffine(image, t, dest_size, flags = cv.inter.cubic, borderMode = cv.border.replicate)
    target = cv.warpAffine(target, t, dest_size, flags = cv.inter.nearest, borderMode = cv.border.constant, borderValue = 255)
    weight = cv.warpAffine(weight, t, dest_size, flags = cv.inter.nearest, borderMode = cv.border.constant, borderValue = 0)

    return {'image':image, 'target':target.long(), 'weight':weight}


def modify(f, key):
    def inner(data):
        assert type(data) is dict and (key in data)

        data = data.copy()
        data[key] = f(data[key])

        return data
    return inner


def size_tensor(t):
    return torch.LongTensor(list(t.size()))



def centre_on(dest_size, background=(0, 0, 0)):
    assert(len(background) == 3)
    assert(len(dest_size) == 2)

    def f(data):
        image = data['image']

        centre = size_tensor(image).float() * 0.5
        toCentre = translation(-centre[0], -centre[1])
        fromCentre = translation(dest_size[0] * 0.5, dest_size[1] * 0.5)
        t = fromCentre.mm(toCentre)

        return warp_affine(data, t, dest_size)
    return f


border_fill = [255 * x for x in default_statistics.mean]



def scale(scale):

    s = transforms.scaling(scale, scale)
    def do_scale(data):
        image = data['image']
        dest_size = (int(image.size(1) * scale), int(image.size(0) * scale))
        return warp_affine(data, s, dest_size, border_fill = border_fill)

    if scale == 1.0:
        return identity
    else:
        return do_scale


def random_crop(input_crop, dest_size, scale_range=(1, 1), rotation_size=0):
    def crop(data):
        t = transforms.make_affine_crop(data['image'].size(), input_crop, dest_size, scale_range, rotation_size)
        return warp_affine(data, t, dest_size, border_fill = border_fill)
    return crop

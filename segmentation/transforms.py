
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

default_statistics = Struct(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def normalize(image, mean=default_statistics.mean, std=default_statistics.mean):
    image = image.float().div_(255)
    for i in range(0, 2):
        image.select(2, i).sub_(mean[i]).div_(std[i])
    return image.permute(0, 3, 1, 2)


def normalize_target(target, num_classes, weights=None):
    mask = target < num_classes
    weights = (weights or 1) * mask.float()

    return target.long() * mask.long(), weights


def warp_affine(data, t, dest_size, border_mode=cv.border.replicate, border_fill=default_statistics.mean):
    image, target, weight = data['image'], data['target'], data['weight']

    border_fill = [x * 255 for x in border_fill]

    image = cv.warpAffine(image, t, dest_size, flags = cv.inter.cubic, borderMode = border_mode, borderValue = border_fill)
    target = cv.warpAffine(target, t, dest_size, flags = cv.inter.nearest, borderMode = cv.border.constant, borderValue = 255)
    weight = cv.warpAffine(weight, t, dest_size, flags = cv.inter.nearest, borderMode = cv.border.constant, borderValue = 0)

    return {'image':image, 'target':target.long(), 'weight':weight}


def warp_perspective(data, t, dest_size, border_mode=cv.border.replicate, border_fill=default_statistics.mean):
    image, target, weight = data['image'], data['target'], data['weight']

    border_fill = [x * 255 for x in border_fill]

    image = cv.warpPerspective(image, t, dest_size, flags = cv.inter.cubic, borderMode = border_mode, borderValue = border_fill)
    target = cv.warpPerspective(target, t, dest_size, flags = cv.inter.nearest, borderMode = cv.border.constant, borderValue = 255)
    weight = cv.warpPerspective(weight, t, dest_size, flags = cv.inter.nearest, borderMode = cv.border.constant, borderValue = 0)

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



def random_transform(size, scale_range=(1, 1), rotation_size=0, perspective_jitter=0, pad=0, flip=True):
    t = transforms.random_affine(size, size, translation=0, scale_range=scale_range, rotation_size=rotation_size, flip=flip)
    if perspective_jitter > 0:
        t = t.mm(transforms.random_perspective_jitter(size, perspective_jitter))

    return transforms.fit_transform(size, t, pad=pad)


def input_size(data):
    image = data['image']
    return (image.size(1), image.size(0))


def crop_augmentation(crop_size, scale_range=(1, 1), rotation_size=0, perspective_jitter=0, pad=0, flip=True, border_mode=cv.border.replicate, border_fill=default_statistics.mean):
    def crop(data):
        t, dest_size = random_transform(input_size(data), scale_range, rotation_size, perspective_jitter, flip=flip, pad=pad)
        x, y = transforms.random_region(dest_size, crop_size)
        t = transforms.translation(-x, -y).mm(t)
        return warp_perspective(data, t, crop_size, border_mode=border_mode, border_fill=border_fill)
    return crop


def fit_augmentation(scale_range=(1, 1), rotation_size=0, perspective_jitter=0, border_mode=cv.border.replicate, flip=True, border_fill=default_statistics.mean, pad=0):
    def crop(data):
        t, dest_size = random_transform(input_size(data), scale_range, rotation_size, perspective_jitter, flip=flip, pad=pad)
        return warp_perspective(data, t, dest_size, border_mode=border_mode, border_fill=border_fill)
    return crop

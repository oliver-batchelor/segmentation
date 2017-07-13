
import torch
import random
import math

from tools import tensor, Struct
import tools.image.cv as cv



def random_check(lower, upper):
    if (lower >= upper):
        return (lower + upper) / 2
    else:
        return random.randint(lower, upper)

def random_region(image, size, border = 0):

    w, h = image.size(1), image.size(0)
    tw, th = size

    x1 = random_check(border, w - tw - border)
    y1 = random_check(border, h - th - border)

    return ((x1 + tw * 0.5, y1 + th * 0.5), (tw * 0.5, th * 0.5))



def scaling(sx, sy):
    return torch.DoubleTensor ([
      [sx, 0, 0],
      [0, sy, 0],
      [0, 0, 1]])

def rotation(a):
    sa = math.sin(a)
    ca = math.cos(a)

    return torch.DoubleTensor ([
      [ca, -sa, 0],
      [sa,  ca, 0],
      [0,   0, 1]])

def translation(tx, ty):
    return torch.DoubleTensor ([
      [1, 0, tx],
      [0, 1, ty],
      [0, 0, 1]])


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


def adjust_channel(image, gamma_range = 0.1):
    gamma = random.uniform(1-gamma_range, 1+gamma_range)
    return cv.adjust_gamma(image, gamma)


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

    image = cv.warpAffine(image, t, dest_size, flags = cv.INTER_CUBIC, borderMode = cv.BORDER_CONSTANT, borderValue = border_fill)
    target = cv.warpAffine(target, t, dest_size, flags = cv.INTER_NEAREST, borderMode = cv.BORDER_CONSTANT)
    weight = cv.warpAffine(weight, t, dest_size, flags = cv.INTER_NEAREST, borderMode = cv.BORDER_CONSTANT)

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

# def random_crop(min_crop, dest_size, max_scale = 1, border = 0, squash_dev = 0.0, rotation_dev = 0):
#     def crop(data):
#         image = data['image']
#
#         base_scale = random.uniform(1, min(max_scale, image.size(1) / dest_size[0], image.size(0) / dest_size[1]))
#         sx, sy = base_scale, base_scale * random.uniform(1-squash_dev, 1+squash_dev)
#
#         crop_size = (math.floor(sx * min_crop[0]), math.floor(sy * min_crop[1]))
#         centre, extents = random_region(image, crop_size, border)
#
#         toCentre = translation(-centre[0], -centre[1])
#         fromCentre = translation(dest_size[0] * 0.5, dest_size[1] * 0.5)
#
#         flip = 1 if random.uniform(0.0, 1.0) > 0.5 else -1
#
#         r = rotation(random.uniform(-rotation_dev, rotation_dev) * (math.pi / 180))
#         s = scaling(flip / sx, 1 / sy)
#         t = fromCentre.mm(s).mm(r).mm(toCentre)
#
#         border_fill = [255 * x for x in default_statistics.mean]
#
#         return warp_affine(data, t, dest_size, border_fill = border_fill)
#     return crop


border_fill = [255 * x for x in default_statistics.mean]

def crop_by(dest_size, centre, extents, scale, rot):
    sx, sy = scale

    toCentre = translation(-centre[0], -centre[1])
    fromCentre = translation(dest_size[0] * 0.5, dest_size[1] * 0.5)

    flip = 1 if random.uniform(0.0, 1.0) > 0.5 else -1

    r = rotation(rot * (math.pi / 180))
    s = scaling(flip * sx, 1 * sy)
    t = fromCentre.mm(s).mm(r).mm(toCentre)

    return lambda data: warp_affine(data, t, dest_size, border_fill = border_fill)


def scale(scale):

    s = scaling(scale, scale)
    def do_scale(data):
        image = data['image']
        dest_size = (int(image.size(1) * scale), int(image.size(0) * scale))
        return warp_affine(data, s, dest_size, border_fill = border_fill)

    if scale == 1.0:
        return identity
    else:
        return do_scale

def clamp(lower, upper, *xs):
    return min(upper, max(lower, *xs))


def random_crop2(input_crop, dest_size, scale_range=(1, 1), rotation_size=0):
    def crop(data):
        image = data['image']

        min_scale = clamp(scale_range[0], scale_range[1], dest_size[0] / image.size(1), dest_size[1] / image.size(0))
        max_scale = scale_range[1] #clamp(scale_range[0], scale_range[1],  image.size(1) / dest_size[0], image.size(0) / dest_size[1])

        print (min_scale, max_scale, dest_size, image.size())

        scale = random.uniform(min_scale, max_scale)

        crop_size = (math.floor(1/scale * input_crop[0]), math.floor(1/scale * input_crop[1]))
        centre, extents = random_region(image, crop_size, 0)

        rotation = random.uniform(-rotation_size, rotation_size)
        return crop_by(dest_size, centre, extents, (scale, scale), rotation)(data)

    return crop

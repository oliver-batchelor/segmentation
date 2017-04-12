
import torch
import random

from tools import tensor

from PIL import Image


def random_region(img, size):
    w, h = img.size
    th, tw = size

    assert(w >= tw and h >= th)

    x1 = random.randint(0, w - tw)
    y1 = random.randint(0, h - th)

    return (x1, y1, x1 + tw, y1 + th)


def random_crop(min_crop, max_crop, input_size, target_size):
    def crop(image, target):
        assert(target.mode == 'L')

        crop_size = (random.randint(min_crop[0], max_crop[0]), random.randint(min_crop[1], max_crop[1]))
        region = random_region(image, crop_size)

        image = image.crop(region).resize(input_size, Image.BICUBIC)
        target = target.crop(region).resize(target_size, Image.NEAREST)

        return tensor.to_tensor_t(image), tensor.to_tensor_t(target)
    return crop

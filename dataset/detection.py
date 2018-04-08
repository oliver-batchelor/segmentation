from os import path
import random

import torch
from torch.utils.data.sampler import RandomSampler
from torch.utils.data import DataLoader

from tools.dataset.flat import FlatList
from tools.dataset.samplers import RepeatSampler
from tools.image import transforms, cv

from tools.image.index_map import default_map
from tools import tensor

from detection import boxes, anchors

def load_boxes(image):
    #print(image)
    img = cv.imread_color(image['file'])
    return {**image, 'image':img}


def random_mean(mean, magnitude):
    return mean + random.uniform(-magnitude, magnitude)

def random_crop(dest_size, scale_range=(1, 1), non_uniform_scale=0, border = 0, min_overlap = 0.5):
    cw, ch = dest_size

    def apply(d):
        scale = random.uniform(*scale_range)
        flip = random.uniform(0, 1) > 0.5

        sx, sy = random_mean(1, non_uniform_scale) * scale, random_mean(1, non_uniform_scale) * scale

        image = d['image']

        input_size = (image.size(1), image.size(0))
        region_size = (cw / sx, ch / sy)

        x, y = transforms.random_region(input_size, region_size, border)
        centre = (x + region_size[0] * 0.5, y + region_size[1] * 0.5)

        t = transforms.make_affine(dest_size, centre, scale=(sx * (-1 if flip else 1), sy))

        boxes = d['boxes']

        if flip:
            boxes = box.transform(boxes, (-region_size[0] -x, -y), (-sx, sy))
        else:
            boxes = box.transform(boxes, (-x, -y), (sx, sy))

        box.clamp(boxes, (0, 0), dest_size)
        boxes, labels = box.filter_invalid(boxes, d['labels'])

        return {**d,
                'image': transforms.warp_affine(image, t, dest_size),
                'boxes': boxes,
                'labels': labels
            }
    return apply

def replace(d, key, value):
    return {**d, key:value}

def over(key, f):
    def modify(d):
        value = f(d[key])
        return replace(d, key, value)
    return modify

def transpose(dicts):
    accum = {}
    for d in dicts:
        for k, v in d.items():
            if k in accum:
                accum[k].append(v)
            else:
                accum[k] = [v]

    return accum


def load_training(args, images):
    return DataLoader(images,
        num_workers=args.num_workers,
        batch_size=1 if args.no_crop else args.batch_size,
        sampler=RepeatSampler(args.epoch_size, len(images)) if args.epoch_size else RandomSampler(images))

def load_testing(args, encoder, images):
    return DataLoader(test, num_workers=args.num_workers, batch_size=1)



def encode_targets(encoder):
    def f(d):
        image = d['image']
        targets = encoder.encode(image, d['labels'], d['boxes'])
        return {
            'image':image,
            'targets': targets
        }
    return f


def transform_training(args, encoder):
    s = 1 / args.down_scale
    result_size = int(args.image_size * s)

    crop = random_crop((result_size, result_size), scale_range = (s * args.min_scale, s * args.max_scale), non_uniform_scale = 0.1)
    adjust_colors = over('image', transforms.adjust_gamma(0.1))

    return transforms.compose (crop, adjust_colors, encode_targets(encoder))

def transform_testing(args):
    s = 1 / args.down_scale
    return transforms.compose (transforms.scale(s))




class DetectionDataset:

    def __init__(self, images, encoder, evaluate=False):
        self.images = images
        self.evaluate = evaluate

    def iter(self, args):

        transform = transform_testing(args) if self.evaluate else transform_training(args, encoder)
        images = FlatList(self.images, loader = load_boxes, transform = transform(args))

        if self.evaluate:
            return load_testing(args, images)
        else:
            return load_training(args, images)

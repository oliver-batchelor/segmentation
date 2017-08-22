import argparse
import inspect
import os.path as path
import os

import torch
import shutil

from torch.utils.data import DataLoader

from tools.image import index_map, cv
from tools import tensor, Struct


from segmentation import transforms, loaders
import json


imagesets = Struct (
    train="Segmentation/train.txt",
    val="Segmentation/val.txt",
    trainval="Segmentation/trainval.txt",
    test="Segmentation/test.txt")



def train(root, loader, transform=None):
    return VOC(root, imagesets.train, loader, transform)

def validation(root, loader, transform=None):
    return VOC(root, imagesets.validation, loader, transform)

def train_validation(root, loader, transform=None):
    return VOC(root, imagesets.train_validation, loader, transform)

class VOC():

    def __init__(self, root, imageset, loader, transform=None):

        self.root = root
        self.transform = transform
        self.loader = loader

        with open(path.join(root, "ImageSets", imageset)) as g:
            base_names = g.read().splitlines()

        masks = path.join(root, "SegmentationClass")
        images = path.join(root, "JPEGImages")

        self.images = [(path.join(images, base + ".jpg"), path.join(masks, base + ".png")) for base in base_names ]



    def __getitem__(self, index):
        image, target = self.loader(*self.images[index])
        if self.transform is not None:
            image, target = self.transform(image, target)

        return image, target

    def __len__(self):
        return len(self.images)





voc_classes = [ 'background', 'aeroplane', 'bicycle',  'bird',     'boat',
                'bottle',    'bus',      'car',      'cat',
                'chair',     'cow',      'diningtable', 'dog',
                'horse',     'motorbike', 'person',  'potted plant',
                'sheep',     'sofa',     'train',    'tv/monitor']

def class_name(i):
    return voc_classes[i] if i < len(voc_classes) else 'ignored'

def classes(args):
    return voc_classes



palette =  [(  0,   0,   0, 0),
            (128,   0,   0, 255),
            (  0, 128,   0, 255),
            (128, 128,   0, 255),
            (  0,   0, 128, 255),
            (128,   0, 128, 255),
            (  0, 128, 128, 255),
            (128, 128, 128, 255),
            ( 64,   0,   0, 255),
            (192,   0,   0, 255),
            ( 64, 128,   0, 255),
            (192, 128,   0, 255),
            ( 64,   0, 128, 255),
            (192,   0, 128, 255),
            ( 64, 128, 128, 255),
            (192, 128, 128, 255),
            (  0,  64,   0, 255),
            (128,  64,   0, 255),
            (  0, 192,   0, 255),
            (128, 192,   0, 255),
            (  0,  64, 128, 255)]


ignore_color=(192, 0, 192, 255)
voc_palette = torch.ByteTensor (ignore_color).expand(256, 4).contiguous()
voc_palette.narrow(0, 0, len(palette)).copy_(torch.ByteTensor(palette))

def to_index(r, g, b):
    return (r // 64) * 16 + (g // 64) * 4 + b // 64

label_map = torch.LongTensor(256).fill_(255)
for label, color in enumerate(palette):
    r, g, b, _ = color
    label_map[to_index(r, g, b)] = label




def to_labels(image):
    image.div_(64)
    indices = image.select(2, 0) * 16 + image.select(2, 1) * 4 + image.select(2, 2) * 1
    return tensor.index(label_map, indices)

load_labels = lambda path: to_labels(cv.imread(path))


def remap_classes(classes):

    mapping = torch.LongTensor(256).fill_(0)
    mapping[255] = 255

    for i in range(0, len(classes)):
        index = classes[i]
        mapping[index] = i

    def remap(labels):
        return tensor.index(mapping, labels)

    return remap


def has_foreground(labels):
    labels = labels * (labels < 255).long()

    return (labels > 0).sum() > 0

def convert(input, subset, output, classes=None):

    if not os.path.isdir(output):
        os.makedirs(output)

    remap = remap_classes(classes)

    def identity(data, target):
        return data, target

    dataset = VOC(input, subset, loader=identity)
    for batch_idx, (data, target) in enumerate(dataset):
        filename = os.path.join(output, os.path.basename(data))
        maskfile =  filename + ".mask"

        labels = load_labels(target)

        labels = remap(labels)
        if has_foreground(labels):
            cv.write(labels.view(*labels.size(), 1), ".png", maskfile)
            shutil.copyfile(data, filename)
            print(filename, labels.size())





def write_config(filename, config):
    with open(filename, 'w') as f:
        json.dump(config, f)

def get_class(c):
    assert c in voc_classes, "invalid class: " + c
    return voc_classes.index(c)


def get_color(name):
    return palette[get_class(name)]

def make_config(class_names, palette):

    def make_class(name):
        return {
            'name':name,
            'color':tuple(get_color(name)),
            'weight':1.0
        }

    class_config = list(map(make_class, class_names))

    return {'classes':class_config, 'ignored':{'id':255, 'color':ignore_color}}



def export_config(output, class_names):
    if not os.path.isdir(output):
        os.makedirs(output)

    config = make_config(["background",  *class_names], palette)
    write_config(os.path.join(output, 'config.json'), config)

def export_voc(input, output, imageset, class_names=voc_classes):



    classes = list(map(get_class,  ["background",  *class_names]))

    if not os.path.isdir(output):
        os.makedirs(output)

    export_config(output,  class_names)
    convert(input, imageset, output, classes=classes)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Pascal VOC, view dataset')

    parser.add_argument('--input', default='/storage/workspace/VOCdevkit/VOC2012',
                        help='input image path')

    parser.add_argument('--imageset', default='trainval',
                        help='voc image subset to use (train|trainval|val|test)')

    parser.add_argument('--output', default=None,
                        help='convert dataset and output to path', required=True)

    parser.add_argument('--restrict', default=None,
                    help='restrict to single label when converting dataset')

    args = parser.parse_args()

    assert args.imageset in imagesets



    class_names = voc_classes

    if args.restrict:
        class_names = args.restrict.split(",")

    export_voc(args.input, args.output, imagesets[args.imageset], class_names)

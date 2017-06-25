import argparse
import inspect
import os.path
import torch
import shutil

from torch.utils.data import DataLoader

from tools.image import transforms, loaders, index_map, cv
from tools import tensor
import tools.dataset.voc as voc
from tools.dataset.samplers import RepeatSampler



voc_classes = [ 'background', 'aeroplane', 'bicycle',  'bird',     'boat',
                'bottle',    'bus',      'car',      'cat',
                'chair',     'cow',      'diningtable', 'dog',
                'horse',     'motorbike', 'person',  'potted plant',
                'sheep',     'sofa',     'train',    'tv/monitor']

def class_name(i):
    return voc_classes[i] if i < len(voc_classes) else 'ignored'

def classes(args):
    return voc_classes



palette =  [(  0,   0,   0),
            (128,   0,   0),
            (  0, 128,   0),
            (128, 128,   0),
            (  0,   0, 128),
            (128,   0, 128),
            (  0, 128, 128),
            (128, 128, 128),
            ( 64,   0,   0),
            (192,   0,   0),
            ( 64, 128,   0),
            (192, 128,   0),
            ( 64,   0, 128),
            (192,   0, 128),
            ( 64, 128, 128),
            (192, 128, 128),
            (  0,  64,   0),
            (128,  64,   0),
            (  0, 192,   0),
            (128, 192,   0),
            (  0,  64, 128)]


def flip_rgb(t):
    d = t.dim() - 1
    return torch.cat([t.narrow(d, 2, 1), t.narrow(d, 1, 1), t.narrow(d, 0, 1)], 1)

voc_palette = torch.ByteTensor ([192, 0, 192]).expand(256, 3).contiguous()
voc_palette.narrow(0, 0, len(palette)).copy_(torch.ByteTensor(palette))

voc_palette = flip_rgb(voc_palette)

def to_index(r, g, b):
    return (r // 64) * 16 + (g // 64) * 4 + b // 64

label_map = torch.LongTensor(256).fill_(255)
for label, color in enumerate(palette):
    label_map[to_index(*color)] = label




def to_labels(image):
    image.div_(64)
    indices = image.select(2, 0) + image.select(2, 1) * 4 + image.select(2, 2) * 16
    return tensor.index(label_map, indices)


load_rgb = loaders.load_cached(loaders.load_rgb)

load_labels = lambda path: to_labels(cv.imread(path))
load_target = loaders.load_cached(load_labels)

def training(args):

    crop_args = {'max_scale':1.5, 'border':10, 'squash_dev':0.1, 'rotation_dev':5}

    if args.no_crop:
        args.batch_size = 1

    crop = transforms.identity if args.no_crop else transforms.random_crop((300, 300), (300, 300), **crop_args)

    dataset = voc.VOC(args.input, voc.imagesets[args.imageset],
        loader=loaders.load_both(load_rgb, load_target),
        transform=transforms.compose([crop, transforms.adjust_colors(gamma_dev=0.1)]))


    loader = DataLoader(dataset, num_workers=args.num_workers,
        batch_size=args.batch_size, sampler=RepeatSampler(args.epoch_size, dataset))

    return loader, dataset


def remap_classes(classes):

    mapping = torch.LongTensor(256).fill_(0)
    mapping[255] = 255 # leave 'ignore' label

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


    dataset = voc.VOC(args.input, subset, loader=transforms.identity)
    for batch_idx, (data, target) in enumerate(dataset):
        filename = os.path.join(output, os.path.basename(data))
        maskfile =  filename + ".mask"

        labels = load_labels(target)

        labels = remap(labels)
        if has_foreground(labels):
            cv.write(labels.view(*labels.size(), 1), ".png", maskfile)
            shutil.copyfile(data, filename)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Pascal VOC, view dataset')

    parser.add_argument('--input', default='/storage/workspace/VOCdevkit/VOC2012',
                        help='input image path')

    parser.add_argument('--imageset', default='trainval',
                        help='voc image subset to use (train|trainval|val|test)')

    parser.add_argument('--no_crop', action='store_true', default=False,
                        help='train always on full size images rather than cropping')

    parser.add_argument('--output', default=None,
                        help='convert dataset and output to path')

    parser.add_argument('--restrict', default=None,
                    help='restrict to single label when converting dataset')

    args = parser.parse_args()

    assert args.imageset in voc.imagesets

    def get_class(c):
        assert c in voc_classes, "invalid class: " + c
        return voc_classes.index(c)

    class_names = ["background", *(args.restrict.split(",") if args.restrict else voc_classes)]
    classes = list(map(get_class, class_names))

    if(args.output):

        if not os.path.isdir(args.output):
            os.makedirs(args.output)

        with open(os.path.join(args.output, 'classes.txt'), 'w') as f:
            f.write('\n'.join(class_names))

        convert(args.input, voc.imagesets[args.imageset], args.output, classes=classes)


    else:
        args.num_workers = 1
        args.epoch_size = 1024
        args.batch_size = 16

        color_map = index_map.make_color_map(255)
        loader, dataset = training(args)

        remap = remap_classes(classes)

        for batch_idx, (data, target) in enumerate(loader):
            #print(index_map.default_map.size(), voc_palette.size())

            target = remap(target)
            image = index_map.overlay_batches(data, target, cols = 4, color_map=voc_palette)

            print(index_map.counts(target, class_names))

            if (cv.display(image.byte()) == 27):
                break

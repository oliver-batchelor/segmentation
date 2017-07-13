import argparse
import inspect
import os.path
import torch
import shutil

from torch.utils.data import DataLoader

from tools.image import index_map, cv
from tools import tensor
import datasets.voc as voc

from segmentation import transforms, loaders



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

class VOC(data.Dataset):

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

load_labels = lambda path: to_labels(cv.imread(path))


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

    def identity(data, target):
        return data, target

    dataset = voc.VOC(args.input, subset, loader=identity)
    for batch_idx, (data, target) in enumerate(dataset):
        filename = os.path.join(output, os.path.basename(data))
        maskfile =  filename + ".mask"

        labels = load_labels(target)
        print(filename, labels.size())

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

import argparse

import os.path
import torch
import shutil
import json

from torch.utils.data import DataLoader

import dataset.masked as dataset

from tools.image import index_map, cv
from tools import tensor


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

def convert(input, output, classes):

    files = dataset.find_files(input)

    if not os.path.isdir(output):
        os.makedirs(output)

    remap = remap_classes(classes)
    files = dataset.find_files(input)

    for f in files:

        input_image = f['image']
        output_image = os.path.join(output, os.path.basename(input_image))
        output_target =  output_image + ".mask"

        labels = dataset.load_target(f['target'])

        labels = remap(labels)
        if has_foreground(labels):
            cv.write(labels.view(*labels.size(), 1), ".png", output_target)
            shutil.copyfile(input_image, output_image)

        print(output_image)

def write_config(filename, config):
    with open(filename, 'w') as f:
        j = json.dumps(config)
        f.write(j)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Dataset splitter')

    parser.add_argument('--input', help='input dataset path')
    parser.add_argument('--output', help='convert dataset and output to path')

    parser.add_argument('--classes', help='classes to keep (comma separated values)', required=True)
    parser.add_argument('--dirs', help='subdirectories to process images', default="test,train")


    args = parser.parse_args()

    assert args.input and args.output and args.classes
    print(args)

    if not os.path.isdir(args.output):
        os.makedirs(args.output)


    if args.classes:
        class_names = ["background", *(args.classes.split(","))]

    dirs = args.dirs.split(",")

    class_map = {}
    config = None

    with open(os.path.join(args.input, "config.json")) as f:
        config = json.load(f)

        for i, c in enumerate(config['classes']):
            name = c['name']
            class_map[name] = i

    def get_class(name):
        assert name in class_map, "invalid class: " + name
        return class_map[name]

    class_ids = list(map(get_class, class_names))

    config['classes'] = [config['classes'][i] for i in class_ids]
    write_config(os.path.join(args.output, "config.json"), config)

    for path in dirs:
        convert(os.path.join(args.input, path), os.path.join(args.output, path), class_ids)



    # convert(args.input, subset, args.output, classes=classes)

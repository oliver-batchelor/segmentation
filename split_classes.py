import argparse
import inspect
import os.path
import torch
import shutil

from torch.utils.data import DataLoader

from tools.image import transforms, loaders, index_map, cv
from tools import tensor

import datasets.seg as dataset


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

def convert(output, files):

    if not os.path.isdir(output):
        os.makedirs(output)

    remap = remap_classes(classes)
    files = dataset.files(input)

    for batch_idx, (data, target) in enumerate(files):
        filename = os.path.join(output, os.path.basename(data))
        maskfile =  filename + ".mask"

        labels = dataset.load_target(target)

        labels = remap(labels)
        if has_foreground(labels):
            cv.write(labels.view(*labels.size(), 1), ".png", maskfile)
            shutil.copyfile(data, filename)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Dataset splitter')

    parser.add_argument('--input', help='input dataset path')
    parser.add_argument('--output', help='convert dataset and output to path')

    parser.add_argument('--classes', help='classes to keep (comma separated values)')
    # parser.add_argument('--keep_missing', action='store_true', default=False,
    #                     help='run without saving outputs for testing')

    args = parser.parse_args()

    assert args.input and args.output and args.classes
    print(args)

    files = masked.find_files(args.input)


    for path in dirs:
        convert(os.path.join(args.input, path), os.path.join(args.output, path), classes)



    # convert(args.input, subset, args.output, classes=classes)

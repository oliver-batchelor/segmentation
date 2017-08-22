import argparse
import csv

import os
import os.path as path


from torch.utils.data import DataLoader

from tools.image import  index_map, cv
from tools.dataset.samplers import RepeatSampler

from segmentation import transforms, loaders, flat

from dataset import masked

def read_classes(path):
    def read_class(row):
        return row[4].split(",")[0]

    with open(path) as g:
        return [read_class(row) for row in csv.reader(g, delimiter = "\t")][1:]




def find_files(input, subset):

    files = []
    annotations = path.join(input, "annotations", subset)
    images = path.join(input, "images", subset)

    for annotfile in os.listdir(annotations):


        base, ext = os.path.splitext(annotfile)
        imagefile = path.join(images, base) + ".jpg"

        if ext.lower() == ".png" and path.isfile(imagefile):
            files.append( {'image':imagefile, 'target':path.join(annotations, annotfile) })

    assert len(files) > 0, "no files found!"
    return files


def dataset(args):

    classes = ['background', *read_classes(os.path.join(args.input, 'objectInfo150.txt'))]

    training_files = find_files(args.input, "training")

    train = masked.training_on(training_files, args)
    test = masked.testing_on(find_files(args.input, "validation"), args)

    return classes, train, test

import os

import itertools
import argparse
import torch

from os import path
import shutil

from tools.image import cv, index_map
from tools import Struct

from dataset import ade, masked

def crop_offset(t, dx, dy):

    up = t.narrow(0, dx, t.size(0) - dx).narrow(1, dy, t.size(1) - dy)
    down = t.narrow(0, 0, t.size(0) - dx).narrow(1, 0, t.size(1) - dy)

    return up, down

def make_borders(target):

    borders = target.clone().zero_()

    for dx, dy in [(0, 1), (1, 0), (1, 1)]:

        b1, b2 = crop_offset(borders, dx, dy)
        t1, t2 = crop_offset(target, dx, dy)

        b1.add_(t1.ne(t2).type_as(b1))
        b2.add_(t1.ne(t2).type_as(b2))

    return borders.clamp_(0, 1)



def export(files, output):
    if not os.path.isdir(output):
        os.makedirs(output)

    for files in files:
        target = masked.load_target(files['target'])
        borders = make_borders(target)

        if args.show:
            image = masked.load_rgb(files['image'])

            overlay = index_map.overlay_labels(image, target)
            cv.display(overlay)

        out_file = path.join(output, path.basename(files['image']))
        shutil.copyfile(files['image'], out_file)

        print(out_file)

        cv.write(target.view(*target.size(), 1), ".png", out_file + ".mask")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Export ADE borders dataset')
    parser.add_argument('--input', default='/storage/workspace/ade', help='input path of the ade')

    parser.add_argument('--output', required=True, default=None, help='output to path')
    parser.add_argument('--show', action='store_true', default=False, help='visualize outputs')

    args = parser.parse_args()

    training_files = ade.find_files(args.input, "training")
    testing_files = ade.find_files(args.input, "validation")

    export(training_files, path.join(args.output, "train"))
    export(testing_files, path.join(args.output, "test"))

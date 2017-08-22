import os

import itertools
import argparse
import torch

from os import path

from tools.image import cv, index_map
from imports import voc, coco

from tools import Struct


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Export training with MSCOCO testing with Pascal VOC')

    parser.add_argument('--coco', default='/storage/workspace/coco_raw',
                        help='input image path')

    parser.add_argument('--no_crop', action='store_true', default=False,
                        help='train always on full size images rather than cropping')


    parser.add_argument('--voc', default='/storage/workspace/VOCdevkit/VOC2012',
                        help='input image path')

    parser.add_argument('--output', default=None, required=True,
                        help='convert dataset and output to path')

    args = parser.parse_args()




    for c in voc.voc_classes[1:]:

        output = path.join(args.output, c.replace(" ", "_"))
        test = path.join(output, "test")
        train = path.join(output, "train")

        voc.export_config(output, [c])

        coco.export_coco(args.coco, train, 'train2014', [coco.to_coco(c)])

        voc.export_voc(args.voc, test, voc.imagesets.val, [c])
        voc.export_voc(args.voc, train, voc.imagesets.train, [c])

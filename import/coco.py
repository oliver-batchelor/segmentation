import os

import shutil
import itertools
import argparse
import torch

from pycocotools.coco import COCO
import numpy as np

from tools.image import cv

voc_classes = [ 'airplane', 'bicycle',  'bird',     'boat',
                'bottle',    'bus',      'car',      'cat',
                'chair',     'cow',      'dining table', 'dog',
                'horse',     'motorcycle', 'person',  'potted plant',
                'sheep',     'couch',     'train',    'tv']


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Pascal VOC, view dataset')

    parser.add_argument('--input', default='/storage/workspace/coco',
                        help='input image path')


    parser.add_argument('--output', default=None, required=True,
                        help='convert dataset and output to path')


    parser.add_argument('--subset', default=None, required=True,
                        help='subset of dataset')

    parser.add_argument('--restrict', default=None,
                    help='restrict to single label when converting dataset')

    parser.add_argument('--voc', action='store_true', default=False,
                        help='use voc subset of classes')


    args = parser.parse_args()


    ann_file = '%s/annotations/instances_%s.json'%(args.input, args.subset)
    coco=COCO(ann_file)


    cat_ids = coco.getCatIds()

    if args.restrict:
        classes = args.restrict.split(",")
        cat_ids = coco.getCatIds(classes)
    elif args.voc:
        cat_ids = coco.getCatIds(voc_classes)


    cats = coco.loadCats(cat_ids)
    class_names = [cat['name'] for cat in cats]


    concat = lambda xs: list(itertools.chain.from_iterable(xs))
    image_ids = concat([coco.getImgIds(catIds=[cat]) for cat in cat_ids])

    print("found images: ", len(image_ids))
    print(class_names)


    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    with open(os.path.join(args.output, 'classes.txt'), 'w') as f:
        f.write('\n'.join(['background', *class_names]))

    def convert(id):
        info = coco.loadImgs(id)[0]
        file_name = info['file_name']

        input_file = '%s/%s/%s'%(args.input, args.subset, file_name)
        output_file = '%s/%s'%(args.output, file_name)

        labels = torch.LongTensor(info['height'], info['width']).zero_()

        for i, cat in enumerate(cat_ids):
            anns = coco.loadAnns(coco.getAnnIds(id, catIds=[cat]))
            for ann in anns:
                mask = torch.from_numpy(coco.annToMask(ann))
                labels.masked_fill_(mask, i + 1)

        print(file_name, labels.size())

        maskfile =  output_file + ".mask"

        cv.write(labels.view(*labels.size(), 1), ".png", maskfile)
        shutil.copyfile(input_file, output_file)


    images = list(map(convert, image_ids))


    #image = 'http://mscoco.org/images/%d'%(img['id']))

import os

import shutil
import itertools
import argparse
import torch

from pycocotools.coco import COCO
import numpy as np

from tools.image import cv, index_map
import imports.voc as voc

from tools import Struct

    # ['person', 'bicycle', 'car', 'motorcycle',
    # 'airplane', 'bus', 'train', 'truck', 'boat',
    # 'traffic light', 'fire hydrant', 'stop sign',
    # 'parking meter', 'bench', 'bird', 'cat', 'dog',
    # 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    # 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
    # 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    # 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    # 'surfboard', 'tennis racket', 'bottle', 'wine glass',
    # 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    # 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    # 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    # 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    # 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    # 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    # 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    # 'hair drier', 'toothbrush']


coco_mapping = {'aeroplane':'airplane', 'diningtable':'dining table', 'motorbike':'motorcycle', 'sofa':'couch', 'tv/monitor':'tv'}

def to_coco(name):
    return coco_mapping[name] if name in coco_mapping else name

cache = {}

def export_coco(input, output, subset, classes=None):
    ann_file = '%s/annotations/instances_%s.json'%(input, subset)

    coco=cache[ann_file] if ann_file in cache else COCO(ann_file)
    cache[ann_file]=coco

    cat_ids = coco.getCatIds(classes) if classes else coco.getCatIds()
    print("{} classes found".format(len(cat_ids)))

    cats = coco.loadCats(cat_ids)
    class_names = [cat['name'] for cat in cats]

    if classes and (not len(classes) == len(class_names)):
         for c in classes:
             if not (c in class_names):
                 print("missing class " + c)


    concat = lambda xs: list(itertools.chain.from_iterable(xs))
    image_ids = concat([coco.getImgIds(catIds=[cat]) for cat in cat_ids])

    print("found images: ", len(image_ids))
    print(classes, class_names)


    #config = voc.make_config(['background', *class_names], index_map.default_map)
    #voc.write_config(os.path.join(output, 'config.json'), config)

    if not os.path.isdir(output):
        os.makedirs(output)



    def convert(id):
        info = coco.loadImgs(id)[0]
        file_name = info['file_name']

        input_file = '%s/%s/%s'%(input, subset, file_name)
        output_file = '%s/%s'%(output, file_name)

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


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Pascal VOC, view dataset')

    parser.add_argument('--input', default='/storage/workspace/coco_raw',
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

    classes = None
    if args.restrict:
        classes = args.restrict.split(",")
    elif args.voc:
        classes = list(map(to_coco, voc.voc_classes[1:]))

    export_coco(args.input, args.output, args.subset, classes)

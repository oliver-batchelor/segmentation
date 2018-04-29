import os

import shutil
import itertools
import argparse
import torch
import json

from pycocotools.coco import COCO
import numpy as np

from tools.image import cv, index_map
# import imports.voc as voc

from tools import Struct

classes = \
    ['person', 'bicycle', 'car', 'motorcycle',
    'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog',
    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush']


# coco_mapping = {'aeroplane':'airplane', 'diningtable':'dining table', 'motorbike':'motorcycle', 'sofa':'couch', 'tv/monitor':'tv'}
#
# def to_coco(name):
#     return coco_mapping[name] if name in coco_mapping else name


def export_coco(input, subset, target_category='Train', class_inputs=None):
    ann_file = '%s/annotations/instances_%s.json'%(input, subset)

    coco=COCO(ann_file)

    cat_ids = coco.getCatIds(class_inputs) if class_inputs else coco.getCatIds()
    print("{} classes found".format(len(cat_ids)))

    cats = coco.loadCats(cat_ids)
    class_names = [cat['name'] for cat in cats]
    class_map = {cat['id']:cat['name'] for cat in cats}

    if classes and (not len(classes) == len(class_names)):
         for c in class_inputs:
             if not (c in class_names):
                 print("missing class " + c)


    concat = lambda xs: list(itertools.chain.from_iterable(xs))
    image_ids = concat([coco.getImgIds(catIds=[cat]) for cat in cat_ids])

    print("found images: ", len(image_ids))
    print(class_names)

    def convert(id):
        info = coco.loadImgs(id)[0]
        file_name = info['file_name']

        input_image = '%s/%s'%(subset, file_name)
        def export_ann(ann):
            x, y, w, h = ann['bbox']
            return {
              'tag':'ObjBox',
              'classId': ann['category_id'],
              'bounds': {
                'lower': [x, y],
                'upper': [x + w, y + h]
              }
            }

        anns = coco.loadAnns(coco.getAnnIds(id, catIds=cat_ids))
        instances = [export_ann(ann) for ann in anns]

        return {
            'imageFile':input_image,
            'imageSize':[info['width'], info['height']],
            'category':target_category,
            'instances':instances
        }

    images = list(map(convert, image_ids))
    return {
        'config': {
            'root':input,
            'extensions':[".jpg"],
            'classes':class_map
        },
        'images':images
    }


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Pascal VOC, view dataset')

    parser.add_argument('--input', default='workspace/coco',
                        help='input image path')

    parser.add_argument('--output', default=None, required=True,
                        help='convert dataset and output to path')


    parser.add_argument('--restrict', default=None,
                    help='restrict to single label when converting dataset')

    # parser.add_argument('--voc', action='store_true', default=False,
    #                     help='use voc subset of classes')


    args = parser.parse_args()

    classes = None
    if args.restrict:
        classes = args.restrict.split(",")

    # elif args.voc:
    #     classes = list(map(to_coco, voc.voc_classes[1:]))

    subsets = [('train2017', 'Train'), ('val2017', 'Test')]
    exports = {subset : export_coco(args.input,  subset = subset, target_category= category, class_inputs = classes) for subset, category in subsets}


    all = {
        'config' : exports['train2017']['config'],
        'images' : sum([subset['images'] for subset in exports.values()], [])
    }


    with open(args.output, 'w') as outfile:
        json.dump(all, outfile, sort_keys=True, indent=4, separators=(',', ': '))

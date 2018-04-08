
import sys
import random
import math

from tools.model import io

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from detection import box, anchors, display
import argparse

from models import models, loss
from tools.image import cv


def random_box(dim, num_classes):
    cx = random.uniform(0, dim[0])
    cy = random.uniform(0, dim[1])

    sx = random.uniform(0.1, 0.2) * dim[0]
    sy = random.uniform(0.1, 0.2) * dim[1]
    return (cx, cy, sx, sy)


if __name__ == '__main__':

    random.seed(0)
    torch.manual_seed(0)

    parser = argparse.ArgumentParser(description='Test model')

    parser.add_argument('--model', action='append', default=[],
                        help='model type and sub-parameters e.g. "unet --dropout 0.1"')

    args = parser.parse_args()

    print(args)

    num_classes = 2
    model_args = {'num_classes':num_classes, 'input_channels':3}

    creation_params = io.parse_params(models, args.model)
    model, encoder = io.create(models, creation_params, model_args)

    print(model)
    io.model_stats(model)

    batches = 4
    dim = (512, 512)

    images = Variable(torch.FloatTensor(batches, 3, dim[1], dim[0]).uniform_(0, 1))
    loc_preds, class_preds = model.cuda()(images.cuda())

    def random_target():
        num_boxes = random.randint(1, 10)
        boxes = torch.Tensor ([random_box(dim, num_classes) for b in range(0, num_boxes)])
        boxes = box.point_form(boxes)
        labels = torch.LongTensor(num_boxes).random_(0, num_classes)

        return encoder.encode(dim, boxes, labels)

    targets = [random_target() for i in range(0, batches)]

    loc_targets = Variable(torch.stack([loc for loc, _ in targets]).cuda())
    class_targets = Variable(torch.stack([classes for _, classes in targets]).cuda())

    print(loss.loss(loc_targets, class_targets, loc_preds, class_preds))

    detections = encoder.decode_batch(images.data, loc_preds.data.zero_(), class_preds.data)

    classes = {}
    for i, (boxes, labels, confs) in zip(images.data, detections):

        i = i.permute(1, 2, 0)
        key = cv.display(display.overlay(i, boxes, labels, confidence=confs))
        if(key == 27):
            break



    #print(boxes)


    #loss = MultiBoxLoss(num_classes)
    #target = (Variable(boxes.cuda()), Variable(labels.cuda()))

    #print(loss(out, target))


import sys
import random
import math

from tools.model import io

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models.detection.loss import MultiBoxLoss
import detection.box_utils as box_utils

import argparse


from models.detection import models


def random_box(num_classes):
    cx = random.uniform(0, 1)
    cy = random.uniform(0, 1)
    
    sx = random.uniform(0.2, 0.9)
    sy = random.uniform(0.2, 0.9)
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
    model = io.create(models, creation_params, model_args)

    print(model)
    io.model_stats(model)

    batches = 4
    x = Variable(torch.FloatTensor(batches, 3, 4, 4).uniform_(0, 1))
    out = model.cuda()(x.cuda())

    #[print(y.size()) for y in out]
    
    num_boxes = random.randint(1, 10)
    boxes = torch.Tensor ([[random_box(num_classes) for i in range(0, num_boxes)] for b in range(0, batches)])
    boxes = box_utils.point_form(boxes.view(-1, 4)).view(boxes.size())
    
    #print(boxes)
    
    labels = torch.LongTensor(batches, num_boxes).random_(0, num_classes - 1)
        
    loss = MultiBoxLoss(num_classes)
    target = (Variable(boxes.cuda()), Variable(labels.cuda()))

    print(loss(out, target))


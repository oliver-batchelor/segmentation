
import sys
from tools.model import io

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import argparse


from models import models

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Test model')

    parser.add_argument('--model', action='append', default=[],
                        help='model type and sub-parameters e.g. "unet --dropout 0.1"')

    args = parser.parse_args()

    print(args)


    model_args = {'num_classes':2, 'input_channels':3}

    creation_params = io.parse_params(models, args.model)
    model = io.create(models, creation_params, model_args)

    print(model)

    io.model_stats(model)

    x = Variable(torch.FloatTensor(4, 3, 500, 500))
    y = model.cuda()(x.cuda())

    print(y.size())

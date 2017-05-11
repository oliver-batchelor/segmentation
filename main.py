from __future__ import print_function
import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import os
import dataset, arguments

from tools import index_map
from tools.loss import dice, one_hot, confusion_matrix, count_elements, confusion_zero

import tools.cv as cv
import models

args = arguments.get_arguments()

args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

loader, dataset, classes = dataset.training(args)
model_dir = args.model_path
start_epoch = 0

model = None

if(args.load):
    model, model_params, start_epoch = models.load(model_dir)
    print("loaded model: ", model_params)


if (model == None):
    model_params = {
        'model': args.model,
        'depth': args.depth,
        'features': args.nfeatures,
        'input_channels': 3,
        'num_classes': len(classes)
    }
    print("creating model: ", model_params)
    model = models.create(model_params)

assert model_params['num_classes'] == len(classes), "number of classes differs in loaded model"

if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)


def loss_nll(output, labels):
    output = F.log_softmax(output)
    target = Variable(labels.cuda() if args.cuda else labels)

    return F.nll_loss(output, target)


def loss_dice(output, labels):
    target = one_hot(labels, len(classes))
    target = Variable(target.cuda() if args.cuda else target)

    return dice(output, target)

def loss_both(output, labels):
    return loss_nll(output, labels) + loss_dice(output, labels)


loss_functions = {
    "nll" : loss_nll,
    "dice"   : loss_dice,
    "both" : loss_both
    }

assert args.loss in loss_functions, "invalid loss function type"
loss_func = loss_functions[args.loss]


def softmax(output):
    _, inds = F.softmax(output).data.max(1)
    return inds.long().squeeze(1).cpu()


def train(epoch):
    confusion_total = confusion_zero(len(classes))
    loss_total = 0
    n_batches = 0

    model.train()

    for batch_idx, (data, labels) in enumerate(loader):

        input = data.permute(0, 3, 1, 2).float()
        input = Variable(input.cuda() if args.cuda else input)

        optimizer.zero_grad()
        output = model(input)

        loss = loss_func(output, labels)

        inds = softmax(output)
        confusion_total = confusion_total + confusion_matrix(inds, labels, len(classes))
        loss_total = loss_total + loss.data[0]
        n_batches = n_batches + 1

        if args.show:
            overlay = index_map.overlay_batches(data, inds)
            if(cv.display(overlay) == 27):
                exit(1)

        loss.backward()
        optimizer.step()

        progress = batch_idx + 1
        if progress % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, progress * len(data), len(loader) * len(data),
                100. * progress / len(loader), loss_total / n_batches))

            print(confusion_total)
            loss_total = 0
            n_batches = 0
            confusion_total = confusion_total.fill_(0)


for epoch in range(start_epoch + 1, start_epoch + args.epochs):

    train(epoch)
    models.save(model_dir, model, model_params, epoch)

    print("scanning dataset...")
    dataset.rescan()

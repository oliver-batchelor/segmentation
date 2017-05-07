from __future__ import print_function
import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import os

import segnet
import unet

import dataset, arguments

from tools import model_io, index_map
from tools.loss import dice, one_hot, confusion_matrix, count_elements


args = arguments.get_arguments()

args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

num_classes = 2

models = {
    "segnet" : segnet,
    "unet"   : unet }

model_dir = args.model

assert args.model in models, "invalid model type"
model = models[args.model].segmenter(num_classes = num_classes, depth = 6)


if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
train_loader, train_dataset = dataset.training(args)


def loss_nll(output, labels):
    output = F.log_softmax(output)
    target = Variable(labels.cuda() if args.cuda else labels)

    return F.nll_loss(output, target)


def loss_dice(output, labels):
    target = one_hot(labels, num_classes)
    target = Variable(target.cuda() if args.cuda else target)

    return dice(output, target)



loss_functions = {
    "nll" : loss_nll,
    "dice"   : loss_dice }

assert args.loss in loss_functions, "invalid loss function type"
loss_func = loss_functions[args.loss]


def softmax(output):
    _, inds = F.softmax(output).data.max(1)
    return inds.long().squeeze(1).cpu()


def train(epoch):
    confusions = torch.LongTensor (num_classes, num_classes).fill_(0)

    model.train()

    for batch_idx, (data, labels) in enumerate(train_loader):
        data = Variable(data.cuda() if args.cuda else data)

        optimizer.zero_grad()
        output = model(data)

        loss = loss_func(output, labels)

        inds = softmax(output)
        confusions = confusions + confusion_matrix(inds, labels, num_classes)


        if args.show:

            overlay = index_map.overlay_batches(data.data.cpu(), inds)
            overlay.show()
            input("next:")

        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader) * len(data),
                100. * batch_idx / len(train_loader), loss.data[0]))

            print(confusions)
            confusions = confusions.fill_(0)


# def test(epoch):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     for data, target in test_loader:
#         if args.cuda:
#             data, target = data.cuda(), target.cuda()
#         data, target = Variable(data, volatile=True), Variable(target)
#         output = model(data)
#         test_loss += F.nll_loss(output, target).data[0]
#         pred = output.data.max(1)[1] # get the index of the max log-probability
#         correct += pred.eq(target.data).cpu().sum()
#
#     test_loss = test_loss
#     test_loss /= len(test_loader) # loss function already averages over batch size
#     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         test_loss, correct, len(test_loader.dataset),
#         100. * correct / len(test_loader.dataset)))




start_epoch = 1

if args.load_model:
    epoch, state = model_io.load(model_dir)
    model.load_state_dict(state)

for epoch in range(start_epoch, start_epoch + args.epochs):

    train(epoch)
    state = (epoch, model.state_dict())

    model_io.save(model_dir, epoch, state)

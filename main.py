
import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import dataset, arguments

from tools import index_map
import tools.cv as cv
import tools.loss as loss

import logger
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
experiment = logger.enumerate_name(args.experiment, os.listdir(args.log_path))

if args.load:
    model, creation_params, start_epoch = models.load(model_dir)
    experiment = creation_params['experiment']
    print("loaded state: ", creation_params)


if model is None:
    model_params = {
        'depth': args.depth,
        'features': args.nfeatures,
        'input_channels': 3,
        'num_classes': len(classes)
    }

    creation_params = {
        'model': args.model,
        'model_params' : model_params,
        'experiment': experiment
    }

    print("creation state: ", creation_params)
    model = models.create(creation_params)

assert creation_params['model_params']['num_classes'] == len(classes), "number of classes differs in loaded model"


log_file = os.path.join(args.log_path, experiment)
log = logger.Logger(log_file)

if args.cuda:
    model.cuda()

#optimizer = optim.Adam(model.parameters(), lr=args.lr)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
loss_func = models.make_loss(args, len(classes))



def softmax(output):
    _, inds = F.softmax(output).data.max(1)
    return inds.long().squeeze(1).cpu()


def train(epoch):
    confusion_total = loss.confusion_zero(len(classes))
    loss_total = 0
    n_batches = 0

    model.train()

    for _, (data, labels) in enumerate(loader):

        input_data = data.permute(0, 3, 1, 2).float()
        input_data = Variable(input_data.cuda() if args.cuda else input_data)

        optimizer.zero_grad()
        output = model(input_data)

        error = loss_func(output, labels)

        inds = softmax(output)
        confusion_total = confusion_total + loss.confusion_matrix(inds, labels, len(classes))
        loss_total = loss_total + error.data[0]
        n_batches = n_batches + 1

        overlay = index_map.overlay_batches(data, inds)
        #log.image("train/segmentation", overlay, epoch)

        if args.show:
            overlay = index_map.overlay_batches(data, inds)
            if cv.display(overlay) == 27:
                exit(1)

        error.backward()
        optimizer.step()


    avg_loss = loss_total / len(loader)
    log.scalar("train/loss", avg_loss, step=epoch)
    log.scalar("train/lr", args.lr, step=epoch)

    log.scalar("train/batch_size", args.batch_size, step=epoch)


    print('Train Epoch: {}\tLoss: {:.6f}'.format(epoch, avg_loss))
    print(confusion_total)



for e in range(start_epoch + 1, start_epoch + args.epochs):

    train(e)
    models.save(model_dir, model, creation_params, e)

    print("scanning dataset...")
    dataset.rescan()


import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import dataset, arguments

from tools.image import index_map
import tools.image.cv as cv
import tools.model.loss as loss

import tools.logger as logger
import models

from tqdm import tqdm

args = arguments.get_arguments()

args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)




loader, dataset, classes = dataset.training(args)
start_epoch = 0

model = None
output_path = os.path.join(args.log, args.name)


model_path = os.path.join(output_path, 'model.pth')

if args.load:
    model, creation_params, start_epoch = models.load(model_path)
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
        'model_params' : model_params
    }



    print("creation state: ", creation_params)
    model = models.create(creation_params)

assert creation_params['model_params']['num_classes'] == len(classes), "number of classes differs in loaded model"

print("working directory: " + output_path)

log = logger.Null()
if not args.dry_run:

    exists = os.path.exists(output_path) and len(os.listdir(output_path)) > 0

    if exists  and (not args.load):
        backup_name = logger.enumerate_name(args.name, os.listdir(output_paths))
        backup_path = os.path.join(output_paths, backup_name)

        print("moving old experiment to: " + backup_path)
        os.rename(output_path, backup_path)

    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    log = logger.Logger(output_path)


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

    for _, (data, labels) in enumerate(tqdm(loader)):

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

    num_labelled = confusion_total.float().sum(1).squeeze(1)
    num_classified = confusion_total.float().sum(0).squeeze(0)
    correct = confusion_total.float().diag()

    precision = correct / num_labelled
    recall = correct / num_classified


    for i in range(1, len(classes)):
        log.scalar("train/classes/" + classes[i] + "/recall", recall[i], step=epoch)
        log.scalar("train/classes/" + classes[i] + "/precision", precision[i], step=epoch)

    total_correct = correct.sum() / confusion_total.sum()
    log.scalar("train/correct", total_correct, step=epoch)






for e in range(start_epoch + 1, start_epoch + args.epochs):

    train(e)

    if (e + 1) % args.save_interval == 0:
        if not args.dry_run:
            models.save(model_path, model, creation_params, e)

        print("scanning dataset...")
        dataset.rescan()

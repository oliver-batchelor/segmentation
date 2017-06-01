
import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import datasets, arguments

from tools.image import index_map, transforms
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


def to_cuda(t):
    return t.cuda() if args.cuda else t

dataset = datasets.create(args)

test_loader, test_data = dataset.testing(args)
train_loader, train_data = dataset.training(args)


classes = dataset.classes(args)
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
        backup_name = logger.enumerate_name(args.name, os.listdir(args.log))
        backup_path = os.path.join(args.log, backup_name)

        print("moving old experiment to: " + backup_path)
        os.rename(output_path, backup_path)

    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    log = logger.Logger(output_path)


model = to_cuda(model)

#optimizer = optim.Adam(model.parameters(), lr=args.lr)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
loss_func = models.make_loss(args, len(classes))



def softmax(output):
    _, inds = F.softmax(output).data.max(1)
    return inds.long().squeeze(1).cpu()



def make_eval():
    confusion_total = loss.confusion_zero(len(classes))
    loss_total = 0

    def f(data, labels):
        nonlocal confusion_total, loss_total
        input_data = transforms.normalize(to_cuda(data)).permute(0, 3, 1, 2)

        output = model(Variable(input_data))
        error = loss_func(output, labels)

        inds = softmax(output)
        confusion_total.add_(loss.confusion_matrix(inds, labels, len(classes)))
        loss_total = loss_total + error.data[0]

        if args.show:
            overlay = index_map.overlay_batches(data, inds)
            if cv.display(overlay) == 27:
                exit(1)

        return error, (confusion_total, loss_total)

    return f, (confusion_total, loss_total)



def summarize(name, totals, epoch):
    confusion, loss = totals

    avg_loss = loss / len(train_loader)
    log.scalar(name + "/loss", avg_loss, step=epoch)
    log.scalar(name + "/lr", args.lr, step=epoch)
    log.scalar(name + "/batch_size", args.batch_size, step=epoch)

    print(name + ' epoch: {}\tLoss: {:.6f}'.format(epoch, avg_loss))
    print(confusion)

    num_labelled = confusion.float().sum(1).squeeze(1)
    num_classified = confusion.float().sum(0).squeeze(0)
    correct = confusion.float().diag()

    precision = correct / num_labelled
    recall = correct / num_classified

    print ("name,   precision,  recall:")
    for i in range(0, precision.size()):
        print(classes[i], precision[i], recall[i])

    n = precisions.size(0)
    print("avg precision:", precision.narrow(0, 1, n).mean(), "avg recall:", recall.narrow(0, 1, n).mean())



    for i in range(1, len(classes)):
        prefix = name + "/classes/" + classes[i]
        log.scalar(prefix + "/recall", recall[i], step=epoch)
        log.scalar(prefix + "/precision", precision[i], step=epoch)

    total_correct = correct.sum() / confusion.sum()
    log.scalar(name + "/correct", total_correct, step=epoch)

def train(epoch):
    eval, totals = make_eval()
    model.train()
    for _, (data, labels) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        error, totals = eval(data, labels)
        error.backward()
        optimizer.step()
    summarize("train", totals, epoch)


def test(epoch):
    eval, totals = make_eval()
    model.eval()
    for _, (data, labels) in enumerate(tqdm(test_loader)):
        error, totals = eval(data, labels)
    summarize("test", totals, epoch)


for e in range(start_epoch + 1, start_epoch + args.epochs):

    train(e)
    test(e)

    if (e + 1) % args.save_interval == 0:
        if not args.dry_run:
            models.save(model_path, model, creation_params, e)

        if train_data.rescan:
            print("scanning dataset...")
            train_data.rescan()

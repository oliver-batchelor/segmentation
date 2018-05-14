import gc
import os
import math
from os import path

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable

import arguments
import dataset as dataset

from tools import Struct
from tools.image import index_map
from segmentation import transforms

import tools.image.cv as cv
import models.loss as loss
from tools.model import io

import evaluate as evaluate


from models import models
import tools.model as m


from tqdm import tqdm



def train(model, loader, eval, optimizer):
    print("training:")
    stats = 0
    model.train()

    with tqdm(total=len(loader) * loader.batch_size) as bar:
        for data in loader:

            optimizer.zero_grad()
            result = eval(data)
            result.error.backward()
            optimizer.step()
            stats += result.statistics

            bar.update(result.statistics.size)

            del result
            gc.collect()



    return stats


def test(model, loader, eval):
    print("testing:")
    stats = 0
    model.eval()
    for data in tqdm(loader):

        result = eval(data)
        stats += result.statistics

        del result
        gc.collect()

    return stats


def test_images(model, files, eval):
    results = []
    model.eval()

    for (image_file, mask_file) in tqdm(files):
        data = dataset.load_rgb(image_file)
        labels = dataset.load_labels(mask_file)

        results.append((image_file, eval(data)))

    return results



def setup_env(args, classes):

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    if args.no_crop:
        args.batch_size = 1
    print(args)

    start_epoch = 0
    best = 0

    model = None
    output_path = args.log
    model_file = path.join(output_path, "model.pth")

    model_args = {'num_classes':len(classes), 'input_channels':3}
    if args.load:
        model, creation_params, start_epoch, best = io.load(models, model_file, model_args)
    else:
        creation_params = io.parse_params(models, args.model)
        model = io.create(models, creation_params, model_args)

    print("model parameters: ", creation_params)

    print("working directory: " + output_path)
    # output_path, logger = l.make_experiment(args.log, args.name, dry_run=args.dry_run, load=args.load)

    model = model.cuda() if args.cuda else model
#    print(model)

    optimizer = optim.SGD(model.parameter_groups(args), lr=args.lr, momentum=args.momentum)
    loss_func = loss.make_loss(args.loss, len(classes), args.cuda)
    eval = evaluate.module(model, loss_func, classes, show=args.show, use_cuda=args.cuda)

    return Struct(**locals())


def main():

    args = arguments.get_arguments()

    classes, train_loader, test_loader = dataset.load(args)
    env = setup_env(args, classes)

    io.model_stats(env.model)

    def adjust_learning_rate(lr):
        for param_group in env.optimizer.param_groups:
            modified = lr * param_group['modifier'] if 'modifier' in param_group else lr
            param_group['lr'] = modified

    def annealing_rate(epoch, max_lr=args.lr, min_lr=args.lr*0.01):
        t = min(1.0, epoch / args.epochs)
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(t * math.pi/2))


    if args.visualize:
        print(env.model)

        sample_input = next(iter(train_loader))['image']
        evaluate.visualize(env.model, sample_input, use_cuda=args.cuda, name=path.join(env.output_path, "model"), format="svg")


    for epoch in range(env.start_epoch + 1, args.epochs + 1):

        lr = annealing_rate(epoch)
        globals = {'lr': lr}

        adjust_learning_rate(lr)
        print("epoch {}, lr {}, best (mean iou) {}".format(epoch, lr, env.best))


        stats = train(env.model, train_loader, env.eval.run, env.optimizer)
        env.eval.summarize("train", stats, epoch, globals=globals)

        stats = test(env.model, test_loader, env.eval.run)
        env.eval.summarize("test", stats, epoch)

        score = env.eval.score(stats)

        if not args.dry_run and score > env.best:
            io.save(env.model_file, env.model, env.creation_params, epoch, score)
            env.best = score


        print("scanning dataset...")
        classes, train_loader, test_loader = dataset.load(args)

if __name__ == '__main__':
    main()

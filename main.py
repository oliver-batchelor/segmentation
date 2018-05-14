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

from tools import Struct
from tools.image import cv

from detection.models import models
from detection.loss import total_bce
from dataset.load import load_file

from evaluate import eval_train, summarize_train, eval_test, summarize_test

from tools.model import io

from trainer import train, test
from tqdm import tqdm

def create_model(output_path, args, classes):
    model_args = {'num_classes':len(classes), 'input_channels':3}
    creation_params = io.parse_params(models, args.model)

    start_epoch, best = 0, 0
    model, encoder = None, None

    if args.load:
        state_dict, creation_params, start_epoch, best = io.load(output_path)
        model, encoder = io.create(models, creation_params, model_args)
        model.load_state_dict(state_dict)
    else:
        model, encoder = io.create(models, creation_params, model_args)

    return model, encoder, creation_params, start_epoch, best


def setup_env(args, classes):

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    print(args)

    output_path = os.path.join(args.log, args.name)
    model, encoder, creation_params, start_epoch, best = create_model(output_path, args, classes)
    loss_func = total_bce

    print("model parameters: ", creation_params)
    print("working directory: " + output_path)
    #output_path, log = logger.make_experiment(args.log, args.name, dry_run=args.dry_run, load=args.load)

    model = model.cuda() if args.cuda else model
#    print(model)

    optimizer = optim.SGD(model.parameter_groups(args, args.fine_tuning), lr=args.lr, momentum=args.momentum)
    return Struct(**locals())


def main():

    args = arguments.get_arguments()

    def var(t):
        if isinstance(t, list):
            return [var(x) for x in t]

        return Variable(t.cuda()) if args.cuda else t


    classes, train_data, test_data = load_file(args.input)
    env = setup_env(args, classes)

    io.model_stats(env.model)

    def adjust_learning_rate(lr):
        for param_group in env.optimizer.param_groups:
            modified = lr * param_group['modifier'] if 'modifier' in param_group else lr
            param_group['lr'] = modified

    def annealing_rate(epoch, max_lr=args.lr, min_lr=args.lr*0.01):
        t = min(1.0, epoch / args.epochs)
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(t * math.pi))


    for epoch in range(env.start_epoch + 1, args.epochs + 1):

        lr = annealing_rate(epoch)
        globals = {'lr': lr}

        adjust_learning_rate(lr)
        print("epoch {}, lr {:.3f}, best (AP[0.5-0.95]) {:.2f}".format(epoch, lr, env.best))

        stats = train(env.model, train_data.train(args, env.encoder), eval_train(env.loss_func, var), env.optimizer)
        summarize_train("train", stats, epoch, globals=globals)

        stats = test(env.model, test_data.test(args), eval_test(env.encoder, var))
        score = summarize_test("test", stats, epoch)

        if not args.dry_run and score >= env.best:
            io.save(env.output_path, env.model, env.creation_params, epoch, score)
            env.best = score


        # print("scanning dataset...")
        # classes, train_loader, test_loader = dataset.load(args)

if __name__ == '__main__':
    main()

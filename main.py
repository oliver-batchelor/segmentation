
import os
from os import path

import torch
import torch.nn.functional as F
import torch.optim as optim
from tuner_utils import yellowfin

from torch.autograd import Variable

import arguments
import dataset as dataset

from tools import Struct
from tools.image import index_map
from segmentation import transforms

import tools.image.cv as cv
import models.loss as loss

import evaluate as evaluate

import tools.logger as l
import models


from tqdm import tqdm



def train(model, loader, eval, optimizer):
    print("training:")
    stats = 0
    model.train()
    for data in tqdm(loader):


        optimizer.zero_grad()
        result = eval(data)
        result.error.backward()
        optimizer.step()
        stats += result.statistics

    return stats


def test(model, loader, eval):
    print("testing:")
    stats = 0
    model.eval()
    for data in tqdm(loader):
        result = eval(data)
        stats += result.statistics

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

    model = None
    output_path = os.path.join(args.log, args.name)
    model_path = os.path.join(output_path, 'model.pth')

    model_params = {'num_classes':len(classes), 'input_channels':3}
    if args.load:
        model, creation_params, start_epoch = models.load(model_path, **model_params)
        print("loaded state: ", creation_params)

        assert model
    else:
        creation_params = models.get_params(args)
        print("creation state: ", creation_params)
        model = models.create(creation_params, **model_params)





    #assert creation_params['model_params']['num_classes'] == len(classes), "number of classes differs in loaded model"

    print("working directory: " + output_path)
    output_path, logger = l.make_experiment(args.log, args.name, dry_run=args.dry_run, load=args.load)


    model = model.cuda() if args.cuda else model
    print(model)

    #optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    #optimizer = yellowfin.YFOptimizer(model.parameters(), lr=args.lr)

    loss_func = loss.make_loss(args.loss, len(classes), args.cuda)
    eval = evaluate.module(model, loss_func, classes, log=logger, show=args.show, use_cuda=args.cuda)

    return Struct(model_path=model_path, output_path=output_path, logger=logger, model=model, loss_func=loss_func,
                  optimizer=optimizer, eval=eval, start_epoch=start_epoch, creation_params=creation_params)


def main():

    args = arguments.get_arguments()

    test_loader, test_data = dataset.testing(args)
    train_loader, train_data = dataset.training(args)

    classes = dataset.classes(args)
    env = setup_env(args, classes)

    if args.visualize:
        print(env.model)

        sample_input = next(iter(train_loader))['image']
        evaluate.visualize(env.model, sample_input, use_cuda=args.cuda, name=path.join(env.output_path, "model"), format="svg")


    for e in range(env.start_epoch + 1, env.start_epoch + args.epochs):

        globals = {'lr':args.lr, 'batch_size':args.batch_size, 'epoch_size':args.epoch_size}

        stats = train(env.model, train_loader, env.eval.run, env.optimizer)
        env.eval.summarize("train", stats, e, globals=globals)

        stats = test(env.model, test_loader, env.eval.run)
        env.eval.summarize("test", stats, e)


        if (e + 1) % args.save_interval == 0:
            if not args.dry_run:
                models.save(env.model_path, env.model, env.creation_params, e)

            print("scanning dataset...")
            train_loader, train_data = dataset.training(args)

if __name__ == '__main__':
    main()

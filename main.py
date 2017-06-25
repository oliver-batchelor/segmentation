
import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import arguments
import datasets.seg as dataset


from tools.image import index_map, transforms
import tools.image.cv as cv
import tools.model.loss as loss

import evaluate as evaluate

import tools.logger as logger
import models

from tqdm import tqdm





def train(model, loader, eval, optimizer):
    stats = 0
    model.train()
    for (data, labels) in tqdm(loader):
        optimizer.zero_grad()
        result = eval(data, labels)
        result.error.backward()
        optimizer.step()
        stats += result.statistics

    return stats


def test(model, loader, eval):
    stats = 0
    model.eval()
    for (data, labels) in tqdm(loader):
        result = eval(data, labels)
        stats += result.statistics

    return stats


def test_images(model, files, eval):
    results = []
    for (image_file, mask_file) in tqdm(files):
        data = dataset.load_rgb(image_file)
        labels = dataset.load_labels(mask_file)

        results.append((image_file, eval(data, labels)))

    return results


def main():
    args = arguments.get_arguments()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    if args.no_crop:
        args.batch_size = 1

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

    print(args)

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
    output_path, logger = make_experiment(args.log, args.name, dry_run=args.dry_run)


    model = model.cuda() if args.cuda else model

    #optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    loss_func = models.make_loss(args, len(classes))

    eval = evaluate.module(model, loss_func, classes, log=logger, show=args.show, use_cuda=args.cuda)


    for e in range(start_epoch + 1, start_epoch + args.epochs):

        globals = {'lr':args.lr, 'batch_size':args.batch_size, 'epoch_size':args.epoch_size}

        stats = train(model, train_loader, eval.run, optimizer)
        eval.summarize("train", stats, e, globals=globals)

        stats = test(model, test_loader, eval.run)
        eval.summarize("test", stats, e)


        if (e + 1) % args.save_interval == 0:
            if not args.dry_run:
                models.save(model_path, model, creation_params, e)

            if train_data.rescan:
                print("scanning dataset...")
                train_data.rescan()

if __name__ == '__main__':
    main()

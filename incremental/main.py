
import os
from os import path

import torch
import dataset as dataset

import dataset.masked as masked

from tools import Struct
from tools.image import index_map, cv

import arguments

from tqdm import tqdm
import main

def to_batch(t):
    return t.expand(1, *t.size())



def split_at(xs, n):
    return xs[:n], xs[n:]

def run(args):


    classes, _, test_loader = dataset.load(args)
    remaining_images = masked.find_files(path.join(args.input, "train"))

    env = main.setup_env(args, classes)

    training_path = path.join(env.output_path, "images")

    inc = 1
    training, remaining_images = split_at(remaining_images, inc)

    adds = 1
    add_interval = 1
    doubling_interval = 2
    for e in range(1,  args.epochs + 1):

        globals = {'lr': env.lr, 'n':len(training)}

        train_loader = dataset.dataloader(args, masked.training_on(training, args))

        stats = main.train(env.model, train_loader, env.eval.run, env.optimizer)
        env.eval.summarize("train", stats, e)

        stats = main.test(env.model, test_loader, env.eval.run)
        env.eval.summarize("test", stats, e)

        if((e % add_interval) == 0 and len(remaining_images) > 0):
            new_images, remaining_images = split_at(remaining_images, inc)
            training += new_images

            if adds % doubling_interval == 0:
                inc *= 2
            adds += 1


        e += 1

    #dataset.training_on(train_files, args)

    #print(len(remaining), len(train))



        # stats = train(env.model, train_loader, env.eval.run, env.optimizer)
        # env.eval.summarize("train", stats, e, globals=globals)
        #
        # stats = test(env.model, test_loader, env.eval.run)
        # env.eval.summarize("test", stats, e)



if __name__ == '__main__':
    args = arguments.get_arguments()
    run(args)

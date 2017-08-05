
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

def test_images(model, files, eval):
    results = []
    stats = 0
    for files in tqdm(files):

        image = to_batch(masked.load_rgb(files['image']))
        labels = to_batch(masked.load_target(files['target']))

        data = {'image':image, 'target':labels, 'weight':torch.ones(labels.size())}
        result = eval(data)
        results.append((files, data, result.prediction.cpu()))

        stats += result.statistics

    return results, stats


def write(image, extension, path):
    result, buf = cv.imencode(extension, image)
    with open(path, 'wb') as file:
        file.write(buf)

def write_weights(output_path, results):
    if not path.isdir(output_path):
        os.makedirs(output_path)

    training = []

    for files, data, result in results:
        weights = (result != data['target']).permute(1, 2, 0).mul(127).add(127)
        weight_file = path.basename(files['image']) + ".weight"
        weight_file = path.join(output_path, weight_file)

        write(weights, ".png", weight_file)
        files = files.copy()
        files['weight'] = weight_file

        training.append(files)
    return training




def split_at(xs, n):
    return xs[:n], xs[n:]

def run():
    args = arguments.get_arguments()

    classes, _, test_loader = dataset.load(args)
    remaining_images = masked.find_files(path.join(args.input, "train"))

    env = main.setup_env(args, classes)

    training_path = path.join(env.output_path, "images")

    initial_size = 1
    increment = 1
    add_interval = 1


    def add_images(images, epoch):
        results, stats = test_images(env.model, images, env.eval.run)
        env.eval.summarize("new", stats, epoch)

        return write_weights(path.join(env.output_path, "weights"), results)

    new_images, remaining_images = split_at(remaining_images, initial_size)
    training = add_images(new_images, 0)

    for e in range(1,  args.epochs):

        train_loader = dataset.dataloader(args, masked.training_on(training, args))

        stats = main.train(env.model, train_loader, env.eval.run, env.optimizer)
        env.eval.summarize("train", stats, e)

        stats = main.test(env.model, test_loader, env.eval.run)
        env.eval.summarize("test", stats, e)

        if((e % add_interval) == 0 and len(remaining_images) > 0):
            new_images, remaining_images = split_at(remaining_images, increment)
            training += add_images(new_images, e)
        e += 1

    #dataset.training_on(train_files, args)

    #print(len(remaining), len(train))



        # stats = train(env.model, train_loader, env.eval.run, env.optimizer)
        # env.eval.summarize("train", stats, e, globals=globals)
        #
        # stats = test(env.model, test_loader, env.eval.run)
        # env.eval.summarize("test", stats, e)



if __name__ == '__main__':
    run()

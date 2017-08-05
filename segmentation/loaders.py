import torch
import tools.image.cv as cv


import os

from torch.multiprocessing import Manager

def load_cached(f):
    assert(f)

    manager = Manager()
    loaded = manager.dict()

    def load(arg):
        modified = os.path.getmtime(arg)

        if((not (arg in loaded)) or modified > loaded[arg]['modified']):
            loaded[arg] = { 'data' : f(arg), 'modified' : modified }

        return loaded[arg]['data']

    return load

def load_labels(path):
    image = cv.imread(path)
    if image.size(2) == 3:
        image = image.narrow(2, 0, 1)

    return image.squeeze(2).long()

def load_weight(path):
    image = cv.imread(path)
    if image.size(2) == 3:
        image = image.narrow(2, 0, 1)

    return image.squeeze(2).float().div_(127)

def load_rgb(path):
    img = cv.imread_color(path)
    return img




def load_masked(input):

    image = load_rgb(input['image'])
    target = load_labels(input['target'])
    weight = load_weight(input['weight']) if 'weight' in input else torch.ones(target.size()).float()

    return {'image':image, 'target':target, 'weight':weight}


def load_split(image='images', target='annotations'):

    image = load_rgb(input['image'])
    target = load_labels(input['target'])
    weight = load_weight(input['weight']) if input['weight'] else torch.ones(target.size()).float()

    return {'image':image, 'target':target, 'weight':weight}

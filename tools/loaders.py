import torch
import tools.cv as cv

import os

from torch.multiprocessing import Manager

def load_cached(f):
    assert(f)

    manager = Manager()
    loaded = manager.dict()
    
    def load(arg):
        modified = os.path.getmtime(arg)

        if((not (arg in loaded)) or modified > loaded[arg]['modified']):
            print(arg, modified)
            loaded[arg] = { 'data' : f(arg), 'modified' : modified }

        return loaded[arg]['data']

    return load

def load_target_channel(channel = 0):
    def load(path):
        return cv.imread(path).select(2, 0)
    return load

def load_rgb(path):
    return cv.imread(path)

def load_both(load_image, load_target):
    assert(load_image and load_target)

    def load(image_path, target_path):
        return load_image(image_path), load_target(target_path)

    return load

import torch
import tools.cv as cv

def load_cached(f):
    assert(f)
    loaded = {}
    def load(arg):
        if(not (arg in loaded)):
            loaded[arg] = f(arg)
        return loaded[arg]
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

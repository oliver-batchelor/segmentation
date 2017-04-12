import torch.utils.data as data

from PIL import Image
import os
import os.path




def mask_loader(path, target):
    return Image.open(path).convert('RGB'), Image.open(target)


def load_cached(f):
    loaded = {}
    def load(arg):
        if(not (arg in loaded)):
            loaded[arg] = f(arg)
        return loaded[arg]
    return load

def load_target(path):
    channels = Image.open(path).split()
    return channels[1]


def cache_loader():
    load_rgb_cached = load_cached(lambda path: Image.open(path).convert('RGB'))
    load_target_cached = load_cached(load_target)


    def load_image(path:str, target:str) -> (Image, Image):
        return load_rgb_cached(path), load_target_cached(target)

    return load_image

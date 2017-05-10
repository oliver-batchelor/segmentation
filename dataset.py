from tools import transforms, loaders

from torch.utils.data import DataLoader
from tools.datasets import FlatFolder

from tools.samplers import RepeatSampler
import os.path

def training(args):

    load_rgb = loaders.load_cached(loaders.load_rgb)
    load_target = loaders.load_cached(loaders.load_target_channel(0))

    path = '/storage/workspace/trees/images'
    with open(os.path.join(path, 'classes.txt')) as g:
        classes = g.readlines()

    dataset = FlatFolder(path,
        loader = loaders.load_both(load_rgb, load_target),
        transform = transforms.random_crop((320, 240), (320, 240), max_scale = 2) )

    loader = DataLoader(dataset, num_workers = args.num_workers,
        batch_size = args.batch_size, sampler = RepeatSampler(1024, dataset))


    return loader, dataset, classes

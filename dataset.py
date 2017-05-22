import os.path
from torch.utils.data import DataLoader

from tools.image import transforms, loaders
from tools.dataset.flat import FlatFolder
from tools.dataset.samplers import RepeatSampler


def training(args):

    load_rgb = loaders.load_cached(loaders.load_rgb)
    load_target = loaders.load_cached(loaders.load_target_channel(0))

    with open(os.path.join(args.input, 'classes.txt')) as g:
        classes = g.readlines()

    dataset = FlatFolder(args.input,
        loader=loaders.load_both(load_rgb, load_target),
        transform=transforms.random_crop((316, 316), (316, 316), max_scale=1.5))

    loader = DataLoader(dataset, num_workers=args.num_workers,
        batch_size=args.batch_size, sampler=RepeatSampler(args.epoch_size, dataset))


    return loader, dataset, classes

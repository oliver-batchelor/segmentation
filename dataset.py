import os.path
from torch.utils.data import DataLoader

from tools.image import transforms, loaders
from tools.dataset.flat import FlatFolder
from tools.dataset.samplers import RepeatSampler


def classes(args):
    with open(os.path.join(args.input, 'classes.txt')) as g:
        return g.readlines()


load_rgb = loaders.load_cached(loaders.load_rgb)
load_target = loaders.load_cached(loaders.load_target_channel(0))

def training(args):

    dataset = FlatFolder(os.path.join(args.input, "train"),
        loader=loaders.load_both(load_rgb, load_target),
        transform=transforms.compose(
            [transforms.random_crop((316, 316), (316, 316), max_scale=1.5),
             transforms.normalize()]))

    loader = DataLoader(dataset, num_workers=args.num_workers,
        batch_size=args.batch_size, sampler=RepeatSampler(args.epoch_size, dataset))


    return loader, dataset


def testing(args):

    dataset = FlatFolder(os.path.join(args.input, "test"),
        loader=loaders.load_both(load_rgb, load_target),
        transform=transforms.normalize())

    loader = DataLoader(dataset, num_workers=args.num_workers, batch_size=1)


    return loader, dataset

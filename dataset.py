from tools import transforms, loaders

from torch.utils.data import DataLoader
from tools.datasets import FlatFolder

from tools.samplers import RepeatSampler


def training(args):

    load_rgb = loaders.load_cached(loaders.load_rgb)
    load_target = loaders.load_cached(loaders.load_target_channel(0))

    dataset = FlatFolder('/storage/workspace/trees/images',

        loader = loaders.load_both(load_rgb, load_target),
        transform = transforms.random_crop((300, 300), (600, 600), (256, 256), (128, 128)) )

    loader = DataLoader(dataset, num_workers = args.num_workers,
        batch_size = args.batch_size, sampler = RepeatSampler(1024, len(dataset)))


    return loader, dataset

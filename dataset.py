from tools import transforms, loaders

from torch.utils.data import DataLoader
from tools.datasets import FlatFolder

from tools.samplers import RepeatSampler


def training(args):

    dataset = FlatFolder('/mnt/Storage/workspace/trees/images',
        loader = loaders.cache_loader(),
        transform = transforms.random_crop((300, 300), (600, 600), (256, 256), (128, 128)) )

    loader = DataLoader(dataset, num_workers = args.num_workers,
        batch_size = args.batch_size, sampler = RepeatSampler(1024, len(dataset)))

    return loader, dataset

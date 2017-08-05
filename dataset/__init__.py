
from dataset import ade, masked
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler

from tools.dataset.samplers import RepeatSampler

def module(args):
    datasets = {
        "masked" : masked,
        "ade"   : ade
        }

    assert args.dataset in datasets, "invalid dataset type"
    return datasets[args.dataset]


def dataset(args):
    return module(args).dataset(args)


def dataloader(args, dataset):
    return DataLoader(dataset, num_workers=args.num_workers,
        batch_size=1 if args.no_crop else args.batch_size, sampler=RepeatSampler(args.epoch_size, dataset) if args.epoch_size else RandomSampler(dataset))


def load(args):
    classes, train, test = dataset(args)

    train_loader = DataLoader(train, num_workers=args.num_workers,
        batch_size=1 if args.no_crop else args.batch_size, sampler=RepeatSampler(args.epoch_size, len(train)) if args.epoch_size else RandomSampler(train))

    test_loader = DataLoader(test, num_workers=args.num_workers, batch_size=1)
    return classes, train_loader, test_loader


def find_files(args, path):
    return module(args).find_files(path)

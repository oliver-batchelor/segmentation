
from dataset import ade, masked
from torch.utils.data import DataLoader

from tools.dataset.samplers import RepeatSampler

def dataset(args):
    datasets = {
        "masked" : masked,
        "ade"   : ade
        }

    assert args.dataset in datasets, "invalid dataset type"
    return datasets[args.dataset].dataset(args)


def load(args):
    classes, train, test = dataset(args)

    train_loader = DataLoader(train, num_workers=args.num_workers,
        batch_size=1 if args.no_crop else args.batch_size, sampler=RepeatSampler(args.epoch_size, train))

    test_loader = DataLoader(test, num_workers=args.num_workers, batch_size=1)

    return classes, train_loader, test_loader

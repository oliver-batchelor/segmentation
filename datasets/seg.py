import argparse
import os.path

from torch.utils.data import DataLoader

from tools.image import  index_map, cv

from segmentation import transforms, loaders
from segmentation.flat import FlatFolder

from tools.dataset.samplers import RepeatSampler


def classes(args):
    with open(os.path.join(args.input, 'classes.txt')) as g:
        return g.read().splitlines()

# load_rgb = loaders.load_cached(loaders.load_rgb)
# load_target = loaders.load_cached(loaders.load_labels)

load_rgb = loaders.load_rgb
load_target = loaders.load_labels

def files(path):
    return FlatFolder(path, loader=transforms.identity)

def training(args):

    crop_args = {'max_scale':1.5, 'border':20, 'squash_dev':0.1, 'rotation_dev':5}
    crop = transforms.identity if args.no_crop else transforms.random_crop((316, 316), (316, 316), **crop_args)

    dataset = FlatFolder(os.path.join(args.input, "train"),
        loader=loaders.load_both(load_rgb, load_target),
        transform=transforms.compose ([crop, transforms.adjust_colors(gamma_dev=0.1)]))

    loader=DataLoader(dataset, num_workers=args.num_workers,
        batch_size=args.batch_size, sampler=RepeatSampler(args.epoch_size, dataset))


    return loader, dataset


def testing(args):

    dataset = FlatFolder(os.path.join(args.input, "test"),
        loader=loaders.load_both(load_rgb, load_target))

    loader = DataLoader(dataset, num_workers=args.num_workers, batch_size=1)
    return loader, dataset

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Segmentation - view data set')
    parser.add_argument('--test', action='store_true', default=False,
                        help='show testing images rather than training')
    parser.add_argument('--input', default='/storage/workspace/trees',
                        help='input image path')
    parser.add_argument('--no_crop', action='store_true', default=False,
                        help='train always on full size images rather than cropping')


    args = parser.parse_args()

    args.num_workers = 1
    args.epoch_size = 1024
    args.batch_size = 1 if args.no_crop else 16


    class_names = classes(args)
    loader, dataset = testing(args) if args.test else training(args)

    for batch_idx, (data, target) in enumerate(loader):
        image = index_map.overlay_batches(data, target, cols = 1 if args.test else 4)
        print(index_map.counts(target, class_names))


        if (cv.display(image.byte()) == 27):
            break

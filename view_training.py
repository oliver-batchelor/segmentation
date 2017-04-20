import argparse
import sys

import dataset
from tools import tensor, index_map
from PIL import Image


parser = argparse.ArgumentParser(description='Tree segmentation - view training set')
parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                    help='input batch size for training (default: 64)')

args = parser.parse_args()

args.num_workers = 1
train_loader, train_dataset = dataset.training(args)

color_map = index_map.make_color_map(255)

for batch_idx, (data, target) in enumerate(train_loader):
    data = tensor.tile_batch(data)
    target = target.view(target.size(0), 1, target.size(1), target.size(2))
    target = tensor.tile_batch(target)


    overlay = index_map.overlay_labels(data, target, color_map)
    overlay.show()

    input("next:")

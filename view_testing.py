import argparse
import sys

import dataset
from tools import tensor, index_map
import tools.cv as cv

parser = argparse.ArgumentParser(description='Tree segmentation - view testing set')
parser.add_argument('--input', default='/storage/workspace/trees/images',
                    help='input image path')

args = parser.parse_args()

args.num_workers = 1
loader, dataset = dataset.testing(args)

color_map = index_map.make_color_map(255)

for batch_idx, (data, target) in enumerate(loader):
    image = index_map.overlay_batches(data, target, cols = 4)

    if (cv.display(image.byte()) == 27):
        break

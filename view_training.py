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

colorizer = index_map.colorizer_t(255)


def overlay_labels(image, labels):
    dim = (image.size(2), image.size(1))

    labels_color = tensor.to_image_t(colorizer(labels)).resize(dim, Image.NEAREST)
    labels = labels.clamp_(0, 1).mul(255)


    mask = tensor.to_image(labels.squeeze(0), 'L').resize(dim, Image.NEAREST)
    image = tensor.to_image_t(image)

    return Image.composite(labels_color, image, mask)



for batch_idx, (data, target) in enumerate(train_loader):
    data = tensor.tile_batch(data)
    target = tensor.tile_batch(target)

    overlay = overlay_labels(data, target)
    overlay.show()

    input("next:")

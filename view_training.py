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
train_loader, train_dataset = dataset.make_loader(args)




def overlay_labels(image, labels):
    colorizer = index_map.colorizer_t(255)
    labels_color = tensor.to_image_t(colorizer(labels))

    labels = labels.clamp_(0, 1).mul(255)

    mask = tensor.to_image(labels.squeeze(0), 'L')
    image = tensor.to_image_t(image)

    return Image.composite(labels_color, image, mask)



for batch_idx, (data, target) in enumerate(train_loader):

    overlay = overlay_labels(tensor.tile_batch(data), tensor.tile_batch(target))
    overlay.show()

    input("next:")

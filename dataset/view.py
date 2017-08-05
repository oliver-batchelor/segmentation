import argparse
import dataset
from tools.image import  index_map, cv

import arguments

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Segmentation - view data set')
    arguments.dataset_args(parser)

    parser.add_argument('--test', action='store_true', default=False,
                        help='show testing images rather than training')

    args = parser.parse_args()

    args.num_workers = 1
    args.epoch_size = 1024
    args.batch_size = 1 if args.no_crop else 16
    args.weight_scale = 1


    class_names, train, test = dataset.load(args)
    loader = test if args.test else train

    print(class_names)
    for data in loader:
        image = index_map.overlay_batches(data['image'], data['target'], cols = 1 if args.test else 4, alpha=0.4)
        print(data['image'].size(), index_map.counts(data['target'], class_names))


        if (cv.display(image.byte()) == 27):
            break


import argparse
import tools.arguments as common

import models


def dataset_args(parser):

    parser.add_argument('--min_scale', type=float, default=1,
                    help='minimum scaling during preprocessing')

    parser.add_argument('--max_scale', type=float, default=1,
                help='maximum scaling during preprocessing')

    parser.add_argument('--rotation', type=float, default=0,
            help='rotation in degrees during preprocessing')

    parser.add_argument('--jitter', type=float, default=0,
            help='perspective jitter (pixels)')

    parser.add_argument('--pad', type=int, default=0,
            help='border pixels to pad preprocessing images (-ve means crop)')


    parser.add_argument('--image_size', type=int, default=440,
                    help='size of patches to train on')

    parser.add_argument('--input', default='/storage/workspace/trees',
                        help='input image path')
    parser.add_argument('--no_crop', action='store_true', default=False,
                        help='train always on full size images rather than cropping')


    parser.add_argument('--model', action='append', default=[],
                        help='model type and sub-parameters e.g. "unet --dropout 0.1"')


    parser.add_argument('--down_scale', type=int, default=1,
                help='down scale of image_size to test/train on')

    parser.add_argument('--dataset', default='masked',
                        help='dataset type options are (masked|ade)')


    parser.add_argument('--train_folder', default='train',
                help='location of training images within dataset (default=train)')

    parser.add_argument('--test_folder', default='test',
                help='location of training images within dataset (default=test)')


    parser.add_argument('--increment', type=int, default=1,
                    help='images to add at each interval')

    parser.add_argument('--add_interval', type=int, default=1,
                help='interval between adding new images')



def add_arguments(parser):
    parser.add_argument('--experiment', default='experiment',
                        help='name for logged experiment on tensorboard (default None)')
    parser.add_argument('--loss', default='jacc',
                        help='loss function type to use. nll|dice|jacc')

    parser.add_argument('--weight_scale', type=float, default=1.0,
                        help='scale for pixel weights (exponent)')

    dataset_args(parser)
    common.add(parser)


parser = argparse.ArgumentParser(description='Tree segmentation')
add_arguments(parser)

def get_arguments():
    return parser.parse_args()

def default_arguments():
    return parser.parse_args([])

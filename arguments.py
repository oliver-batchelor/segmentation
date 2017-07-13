
import argparse
import tools.arguments as common

import models


def dataset_args(parser):

    parser.add_argument('--min_scale', type=int, default=0.8,
                    help='minimum scaling during preprocessing')

    parser.add_argument('--max_scale', type=int, default=1.2,
                help='maximum scaling during preprocessing')

    parser.add_argument('--rotation', type=int, default=5,
            help='rotation in degrees during preprocessing')

    parser.add_argument('--image_size', type=int, default=360,
                    help='size of patches to train on')

    parser.add_argument('--input', default='/storage/workspace/trees',
                        help='input image path')
    parser.add_argument('--no_crop', action='store_true', default=False,
                        help='train always on full size images rather than cropping')


    parser.add_argument('--down_scale', type=int, default=1,
                help='down scale of image_size to test/train on')

    parser.add_argument('--dataset', default='masked',
                        help='dataset type options are (masked|ade)')
    


def add_arguments(parser):
    parser.add_argument('--experiment', default='experiment',
                        help='name for logged experiment on tensorboard (default None)')
    parser.add_argument('--loss', default='dice',
                        help='loss function type to use. nll|dice|jacc')

    parser.add_argument('--weight_scale', type=float, default=1.0,
                        help='scale for pixel weights (exponent)')

    dataset_args(parser)

    common.add(parser)
    models.add_arguments(parser)


parser = argparse.ArgumentParser(description='Tree segmentation')
add_arguments(parser)

def get_arguments():
    return parser.parse_args()

def default_arguments():
    return parser.parse_args([])

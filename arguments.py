
import argparse
import tools.arguments as common

import models


def dataset_args(parser):

    parser.add_argument('--min_scale', type=float, default=0.75,
                    help='minimum scaling during preprocessing')

    parser.add_argument('--max_scale', type=float, default=1.5,
                help='maximum scaling during preprocessing')




    parser.add_argument('--gamma', type=float, default=0.1,
                help='variation in gamma (brightness) when training')

    parser.add_argument('--image_size', type=int, default=440,
                    help='size of patches to train on')

    parser.add_argument('--input', required=True,
                        help='input image path')
    parser.add_argument('--no_crop', action='store_true', default=False,
                        help='train always on full size images rather than cropping')


    parser.add_argument('--model', action='append', default=[],
                        help='model type and sub-parameters e.g. "unet --dropout 0.1"')


    parser.add_argument('--down_scale', type=int, default=1,
                help='down scale of image_size to test/train on')

    parser.add_argument('--dataset', default='annotate',
                        help='dataset type options are (annotate)')


def add_arguments(parser):
    parser.add_argument('--experiment', default='experiment',
                        help='name for logged experiment on tensorboard (default None)')

    dataset_args(parser)
    common.add(parser)


parser = argparse.ArgumentParser(description='Object detection')
add_arguments(parser)

def get_arguments():
    return parser.parse_args()

def default_arguments():
    return parser.parse_args([])


import argparse
import tools.arguments as common

import models

def get_arguments():
    parser = argparse.ArgumentParser(description='Tree segmentation')



    # Model parameters

    parser.add_argument('--experiment', default='experiment',
                        help='name for logged experiment on tensorboard (default None)')
    parser.add_argument('--loss', default='dice',
                        help='loss function type to use. nll|dice|jacc')


    parser.add_argument('--input', default='/storage/workspace/trees',
                        help='input image path')
    parser.add_argument('--no_crop', action='store_true', default=False,
                        help='train always on full size images rather than cropping')



    common.add(parser)
    models.add_arguments(parser)

    return parser.parse_args()

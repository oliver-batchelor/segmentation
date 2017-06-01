
import argparse
import tools.arguments as common

def get_arguments():
    parser = argparse.ArgumentParser(description='Tree segmentation')



    # Model parameters
    parser.add_argument('--model', default='segnet',
                        help='model type to use. segnet|unet|unet_full')

    parser.add_argument('--dataset', default='seg',
                        help='dataset to load. segnet|voc')

    parser.add_argument('--experiment', default='experiment',
                        help='name for logged experiment on tensorboard (default None)')
    parser.add_argument('--loss', default='dice',
                        help='loss function type to use. nll|dice')
    parser.add_argument('--depth', type=int, default=4,
                        help='number of layers of depth in the model')
    parser.add_argument('--nfeatures', type=int, default=8,
                        help='number of features present in the first layer of the network')

    parser.add_argument('--dropout', type=float, default=0, help='dropout to use at each convolution')

    parser.add_argument('--input', default='/storage/workspace/trees',
                        help='input image path')
    parser.add_argument('--no_crop', action='store_true', default=False,
                        help='train always on full size images rather than cropping')



    common.add(parser)

    return parser.parse_args()

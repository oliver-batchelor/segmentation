
import argparse
import torch


def get_arguments():
    parser = argparse.ArgumentParser(description='Tree segmentation')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--model', default='segnet',
                        help='model type to use. segnet|unet')

    parser.add_argument('--loss', default='dice',
                    help='loss function type to use. nll|dice')
    parser.add_argument('--depth', type=int, default=4,
                    help='number of layers of depth in the model')
    parser.add_argument('--nfeatures', type=int, default=8,
                    help='number of features present in the first layer of the network')

    parser.add_argument('--model_path', default='models',
                    help='path to store models')

    parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--load', action = 'store_true', default=False,
                        help='load progress from previous training')

    parser.add_argument('--show', action = 'store_true', default=False,
                        help='view training output')

    parser.add_argument('--num-workers', type=int, default=1, metavar='W',
                        help='number of workers used to process dataset')

    return parser.parse_args()

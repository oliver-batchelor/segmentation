from __future__ import print_function

import argparse

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from tools.image import index_map, cv, loaders

import models

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tree segmentation test')
    parser.add_argument('image', help='image to load')

    parser.add_argument('--save', help='save result to file with name')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    args = parser.parse_args()

    def softmax(output):
        _, inds = F.softmax(output).data.max(1)
        return inds.long().squeeze(1).cpu()

    def write(image, extension, path):
        result, buf = cv.imencode(extension, image)
        with open(path, 'wb') as file:
            file.write(buf)


    model, model_params, epoch = models.load('log/segmenter/model.pth')
    print("loaded model: ", model_params)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        model.cuda()

    image = loaders.load_rgb(args.image)

    data = image.view(1, *image.size())


    data_input = data.permute(0, 3, 1, 2).float()
    if args.cuda:
        data_input = data_input.cuda()

    model.eval()
    output = model(Variable(data_input))

    inds = softmax(output)

    if args.save:
        inds = inds.squeeze(0)
        labels = inds.view(*inds.size(), 1)

        labels = cv.resize(labels, (image.size(1), image.size(0)), interpolation=cv.INTER_NEAREST)
        write(labels, ".png", args.save)
    else:
        overlay = index_map.overlay_batches(data, inds)
        cv.display(overlay)

#if(args.show):

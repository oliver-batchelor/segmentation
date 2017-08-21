from __future__ import print_function

import argparse
from os import path
import os

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from tools.model import io

from tools.image import index_map, cv
from segmentation import loaders, transforms, flat

import models as m

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tree segmentation test')
    parser.add_argument('--image', default=None, help='image to load')
    parser.add_argument('--batch', default=None, help='path to load all images')


    parser.add_argument('--model', default='trees/pretrained/model.pth', help='path of model to use')
    parser.add_argument('--save', help='save result to file with name (or directory with name)')

    parser.add_argument('--show', action='store_true', default=False, help='display outputs')


    args = parser.parse_args()

    assert (args.image or image.path) and not (args.image and image.path), "required: image filename or path for batch processing"

def softmax(output, dim=1):
    _, inds = F.softmax(output).data.max(dim)
    return inds.long().cpu()

def write(image, extension, path):
    result, buf = cv.imencode(extension, image)
    with open(path, 'wb') as file:
        file.write(buf)


model_args = {'num_classes':2, 'input_channels':3}
model, creation_params, start_epoch, best = io.load(m.models, model_path, model_args)

print("loaded model: ", creation_params)

model.cuda()
model.eval()



def eval(image_path, save=False, show=False):
    image = loaders.load_rgb(image_path)
    data = image.view(1, *image.size())

    data_input = transforms.normalize(data)
    data_input = data_input.cuda()


    output = model(Variable(data_input))

    inds = softmax(output).squeeze(0)
    labels = inds.view(*inds.size(), 1)

    overlay = index_map.overlay_labels(image, labels)

    if save:
        if not os.path.isdir(save):
            os.makedirs(save)

        labels = cv.resize(labels, (image.size(1), image.size(0)), interpolation=cv.inter.nearest)
        write(labels, ".png", path.join(save, "labels.png"))
        write(overlay, ".jpg", path.join(save, "overlay.jpg"))


        output = output.data.cpu().squeeze(0)
        norm_output = F.softmax(output).data.cpu().squeeze(0)

        for i in range(0, output.size(0)):
            norm_features = norm_output[i] * 255
            write(features.view(*features.size(), 1), ".jpg", path.join(save, "class{}.jpg".format(i)))


        print("wrote outputs into directory: " + save)

    if show:
        cv.display(overlay)


if args.image:
    eval(args.image, args.save, args.show)
elif args.batch:
    files = flat.find_files(args.batch, flat.image_file)

    for f in files:
        base = os.path.basename(f)
        save = os.path.join(args.save, base) if args.save or None
        eval(f, save, args.show)

from __future__ import print_function

import os
import shutil

import torch
import argparse

import tools.dataset.flat as flat



parser = argparse.ArgumentParser(description='Tree segmentation test')
parser.add_argument('input_path', help='input path to read dataset for split')
parser.add_argument('output_path', help='path to write test/train splits')

parser.add_argument('n', type=int, help='number of images to use for testing', default='None')
args = parser.parse_args()


test_dir = os.path.join(args.output_path, 'test')
train_dir = os.path.join(args.output_path, 'train')
os.makedirs(test_dir)
os.makedirs(train_dir)

images = flat.find_files(args.input_path,  flat.image_with_mask(flat.imageExtensions))

assert len(images) > args.n, "not enough images found for partition of size " + n
indices = torch.randperm(len(images))

def files_from(start, n):
    return sum ([list(images[i]) for i in indices.narrow(0, start, n)], [])

test_files = files_from(0, args.n)
train_files = files_from(args.n, len(images) - args.n)

for f in test_files:
    shutil.copy(f, test_dir)

for f in train_files:
    shutil.copy(f, train_dir)

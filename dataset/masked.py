
import os.path
import json
import torch


from segmentation import transforms, loaders, flat



load_rgb = loaders.load_rgb
load_target = loaders.load_labels

def read_config(filename):
    with open(filename) as f:
        config = json.load(f)
        class_names = [c['name'] for c in config['classes']]
        colors      =  [c['color'] for c in config['classes']]
        weights = [c['weight'] for c in config['classes']]

        palette = torch.ByteTensor(255, 4).fill_(0)
        palette.narrow(0, 0, len(colors)).copy_(torch.ByteTensor(colors))

        return class_names, colors, torch.FloatTensor(weights)

    assert false



def training_on(files, args):

    s = 1 / args.down_scale
    random_args = {'scale_range':(s * args.min_scale, s * args.max_scale), 'rotation_size': args.rotation, 'perspective_jitter' : args.jitter, 'pad':args.pad}

    result_size = int(args.image_size * s)
    crop = transforms.fit_augmentation(**random_args) if args.no_crop else transforms.crop_augmentation((result_size, result_size), **random_args)

    return flat.FileList(files,
        loader=loaders.load_masked,
        transform=transforms.compose ([transforms.scale_weights(args.weight_scale), crop, transforms.adjust_colors(gamma_range = 0.1)]))

def testing_on(files, args):
    s = 1 / args.down_scale

    return flat.FileList(files,   loader=loaders.load_masked, transform=transforms.scale(s))

def find_files(path):
    return flat.find_files(path, flat.image_with_mask(flat.image_extensions))


def dataset(args):


    class_names, _, _ = read_config(os.path.join(args.input, 'config.json'))

    train = training_on(find_files(os.path.join(args.input, args.train_folder)), args)
    test = testing_on(find_files(os.path.join(args.input, args.test_folder)), args)

    return class_names, train, test


import os.path


from segmentation import transforms, loaders, flat



load_rgb = loaders.load_rgb
load_target = loaders.load_labels

def read_classes(filename):
    with open(filename) as g:
        return g.read().splitlines()



def training_on(files, args):
    s = 1 / args.down_scale
    crop_args = {'scale_range':(s * args.min_scale, s * args.max_scale), 'rotation_size': args.rotation}

    result_size = int(args.image_size * s)
    crop = transforms.scale(s)  if args.no_crop else transforms.random_crop2((args.image_size, args.image_size), (result_size, result_size), **crop_args)

    return flat.FileList(files,
        loader=loaders.load_masked,
        transform=transforms.compose ([transforms.scale_weights(args.weight_scale), crop, transforms.adjust_colors(gamma_range = 0.1)]))

def testing_on(files, args):
    s = 1 / args.down_scale

    return flat.FileList(files,   loader=loaders.load_masked, transform=transforms.scale(s))




def dataset(args):

    def find_files(path):
        return flat.find_files(path, flat.image_with_mask(flat.image_extensions))


    classes = read_classes(os.path.join(args.input, 'train', 'classes.txt'))

    train = training_on(find_files(os.path.join(args.input, "train")), args)
    test = testing_on(find_files(os.path.join(args.input, "test")), args)

    return classes, train, test

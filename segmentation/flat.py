import torch.utils.data as data
import os
import os.path


image_extensions = ['jpg', 'jpeg', 'png', 'ppm', 'bmp']

def has_extension(extensions, filename):
    return any(filename.lower().endswith("." + extension) for extension in extensions)


def file_extension(extensions):
    def f(filename):
        if has_extension(extensions, filename):
            return filename
    return f

image_file = file_extension(image_extensions)

def image_with_mask(extensions):
    def f(filename):
        mask_filename = filename + ".mask"
        if(has_extension(extensions, filename) and os.path.exists(mask_filename)):
            return (filename, mask_filename)
    return f


def file_with_annotations(extensions, annotations):
    def f(filename):
        files = {a : mask_filename + "." + a for a in annotations}
        if(has_extension(extensions, filename) and all(map(os.path.exists), files.values())):
            return (filename, files)
    return f

def find_files(dir, file_filter):
    images = []
    for fname in os.listdir(dir):
        item = file_filter(os.path.join(dir, fname))
        if(item):
            images.append(item)

    return images


class FlatFolder(data.Dataset):

    def __init__(self, root, loader, file_filter = image_with_mask(image_extensions), transform=None):

        self.root = root
        self.transform = transform
        self.loader = loader
        self.file_filter = file_filter

        self.rescan()


    def rescan(self):
        self.images = find_files(self.root, self.file_filter)
        if len(self.images) == 0:
            raise(RuntimeError("Found 0 matching images in: " + self.root + "\n"))



    def __getitem__(self, index):

        image, target = self.loader(*self.images[index])
        if self.transform is not None:
            image, target = self.transform(image, target)

        return image, target

    def __len__(self):
        return len(self.images)

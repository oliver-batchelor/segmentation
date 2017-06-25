import torch.utils.data as data

from tools import Struct
import os.path as path

imagesets = Struct (
    train="Segmentation/train.txt",
    val="Segmentation/val.txt",
    trainval="Segmentation/trainval.txt",
    test="Segmentation/test.txt")



def train(root, loader, transform=None):
    return VOC(root, imagesets.train, loader, transform)

def validation(root, loader, transform=None):
    return VOC(root, imagesets.validation, loader, transform)

def train_validation(root, loader, transform=None):
    return VOC(root, imagesets.train_validation, loader, transform)

class VOC(data.Dataset):

    def __init__(self, root, imageset, loader, transform=None):

        self.root = root
        self.transform = transform
        self.loader = loader

        with open(path.join(root, "ImageSets", imageset)) as g:
            base_names = g.read().splitlines()

        masks = path.join(root, "SegmentationClass")
        images = path.join(root, "JPEGImages")

        self.images = [(path.join(images, base + ".jpg"), path.join(masks, base + ".png")) for base in base_names ]



    def __getitem__(self, index):
        image, target = self.loader(*self.images[index])
        if self.transform is not None:
            image, target = self.transform(image, target)

        return image, target

    def __len__(self):
        return len(self.images)


import cv2
import torch

def imread(path):
    image = torch.from_numpy (cv2.imread(path, cv2.IMREAD_UNCHANGED))
    if(image.dim() == 2):
        image = image.view(*image.size(), 1)
    return image

def imencode(extension, image):
    assert(image.dim() == 3 and image.size(2) <= 4)
    return cv2.imencode(extension, image.numpy())

def imwrite(path, image):
    assert(image.dim() == 3 and image.size(2) <= 4)
    return cv2.imwrite(path, image.numpy())

waitKey = cv2.waitKey

def display(t):
    imshow("image", t)
    return waitKey()


def imshow(name, t):
    cv2.imshow(name, t.numpy())



def warpAffine(image, t, target_size, **kwargs):
    t = t.narrow(0, 0, 2)
    return torch.from_numpy(cv2.warpAffine(image.numpy(), t.numpy(), target_size, **kwargs))

def resize(image, dim, **kwargs):
    channels = image.size(2)
    result = torch.from_numpy(cv2.resize(image.numpy(), dim, **kwargs))
    return result.view(dim[1], dim[0], channels)

INTER_CUBIC = cv2.INTER_CUBIC
INTER_NEAREST = cv2.INTER_NEAREST

BORDER_REPLICATE = cv2.BORDER_REPLICATE
BORDER_CONSTANT = cv2.BORDER_CONSTANT


#BORDER_REPEAT = cv2.BORDER_REPEAT

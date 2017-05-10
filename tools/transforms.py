
import torch
import random
import math

from tools import tensor
import tools.cv as cv

def random_region(image, size):

    w, h = image.size(1), image.size(0)
    th, tw = size

    assert(w >= tw and h >= th)

    x1 = random.randint(0, w - tw)
    y1 = random.randint(0, h - th)

    return ((x1 + tw * 0.5, y1 + th * 0.5), (tw * 0.5, th * 0.5))



def scaling(sx, sy):
    return torch.DoubleTensor ([
      [sx, 0, 0],
      [0, sy, 0],
      [0, 0, 1]])

def rotation(a):
    sa = math.sin(a)
    ca = math.cos(a)

    return torch.DoubleTensor ([
      [ca, -sa, 0],
      [sa,  ca, 0],
      [0,   0, 1]])

def translation(tx, ty):
    return torch.DoubleTensor ([
      [1, 0, tx],
      [0, 1, ty],
      [0, 0, 1]])




# def adjust_gamma(image, gamma=1.0):
#
# 	invGamma = 1.0 / gamma
# 	table = np.array([((i / 255.0) ** invGamma) * 255
# 		for i in np.arange(0, 256)]).astype("uint8")
#
# 	return cv2.LUT(image, table)

def random_crop(min_crop, dest_size, max_scale = 1):
    def crop(image, target):

        scale = random.uniform(1, min(max_scale, image.size(1) / dest_size[0], image.size(0) / dest_size[1]))

        crop_size = (math.floor(scale * min_crop[0]), math.floor(scale * min_crop[1]))
        centre, extents = random_region(image, crop_size)

        toCentre = translation(-centre[0], -centre[1])
        fromCentre = translation(dest_size[0] * 0.5, dest_size[1] * 0.5)

        r = rotation(random.uniform(-10, 10) * (math.pi / 180))
        s = scaling(1 / scale, 1 / scale)
        t = fromCentre.mm(s).mm(r).mm(toCentre)

        image_out = cv.warpAffine(image, t, dest_size, flags = cv.INTER_CUBIC, borderMode = cv.BORDER_CONSTANT)
        labels_out = cv.warpAffine(target, t, dest_size, flags = cv.INTER_NEAREST, borderMode = cv.BORDER_CONSTANT)

        return image_out, labels_out.long()
    return crop

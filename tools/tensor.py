import torch
from tools import index_map

from PIL import Image

def to_tensor(pic):

    assert(isinstance(pic, Image.Image))
    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    return img


def to_tensor_t(pic):
    return to_tensor(pic).permute(2, 0, 1)

def to_image(pic, mode=None):
    assert(isinstance(pic, torch.ByteTensor))

    print(pic.numpy().shape)

    return Image.fromarray(pic.numpy())

def to_image_t(pic, mode=None):
    return to_image(pic.permute(1, 2, 0), mode)


def split(t, dim = 0):
    return [chunk.squeeze(dim) for chunk in t.chunk(t.size(dim), dim)]


def tile_batch(t, cols = int(6)):
    assert(t.dim() == 4)

    h = t.size(2)
    w = t.size(3)

    rows = (t.size(0) - 1) // cols + 1
    out = t.new(t.size(1), rows * h, cols * w).fill_(0)

    for i in range(0, t.size(0)):
        x = i % cols
        y = i // cols

        tile = out.narrow(2, x * w, w).narrow(1, y * h, h)
        tile.copy_(t[i])

    return out

def show_batch(t, cols = int(6)):
    tiled = tile_batch(t, cols)
    to_image_t(tiled).show()

def show_indexed_batch(t, cols = int(6)):
    colorizer = index_map.colorizer_t(255)
    tiled = tile_batch(t, cols)

    color = colorizer(tiled)
    to_image_t(color).show()

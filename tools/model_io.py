import os
import sys
import torch

def save(path, epoch, state):
    if(not os.path.isdir(path)):
        os.mkdir(path)

    epoch_name = 'epoch_%d.pth' % epoch
    filename = os.path.join(path, epoch_name)
    print('saving %s' % filename)

    torch.save(state, filename)

    model_file = os.path.join(path, 'model.pth')
    if(os.path.isfile(model_file)):
        os.remove(model_file)

    os.symlink(epoch_name, model_file)

def load(path):
    model_file = os.path.join(path, 'model.pth')
    print('loading model %s' % model_file)

    return torch.load(model_file)

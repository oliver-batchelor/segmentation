import math
import torch

from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init

import models.unet as model
from tools import Struct, tensor


def add_arguments(parser):
    for name, (default, help) in model.parameters.items():
        parser.add_argument('--' + name, default=default, type=type(default), help=help)


def get_params(args):
    params = {}

    for name in model.parameters.keys():
        params[name] = getattr(args, name)

    return Struct(**params)


def create(params, **args):
    return model.create(params, **args)

def save(path, model, model_params, epoch):
    state = {
        'epoch': epoch,
        'params': model_params,
        'state': model.state_dict()
    }

    torch.save(state, path)

def load(path, **args):

    state = torch.load(path)
    params = state['params']
    model = create(params, **args)

    model.load_state_dict(state['state'])

    return model, params, state['epoch']

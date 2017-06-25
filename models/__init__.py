from models import unet, unet_full, segnet

import tools.model.loss as loss
from tools import Struct, tensor


import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

import torch.nn.init as init
import math

models = {
    "unet"   : unet,
    }

def without(d, key):
    new_d = d.copy()
    new_d.pop(key)
    return new_d

def create(params):
    model_type = params['model']

    assert model_type in models, "invalid model type"
    model = models[model_type].segmenter(**params['model_params'])

    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            init.xavier_uniform(m.weight.data, gain=math.sqrt(2))
            init.constant(m.bias.data, 0.1)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    return model

def save(path, model, model_params, epoch):
    state = Struct(epoch=epoch, params=model_params, state=model.state_dict())

    print("saving state to: " + path)
    torch.save(state, path)

def load(path):

    state = torch.load(path)
    params = state.params
    model = create(params)

    model.load_state_dict(state.state)
    return model, params, state.epoch


def make_loss(args, num_classes):

    def var(labels):
        return Variable(labels.cuda() if args.cuda else labels)

    def loss_nll(output, labels, weights):
        return F.nll_loss(F.log_softmax(output), var(labels))


    def loss_dice(output, labels, weights=None):
        target = tensor.one_hot(labels, num_classes)
        return loss.dice(output, var(target), var(weights) if weights else None)

    def loss_jacc(output, labels, weights=None):
        target = tensor.one_hot(labels, num_classes)
        return loss.jacc(output, var(target), var(weights) if weights else None)

    def loss_iou(output, labels, weights=None):
        target = tensor.one_hot(labels, num_classes)
        return loss.iou(output, var(target), var(weights) if weights else None)


    loss_functions = {
        "nll" : loss_nll,
        "dice"   : loss_dice,
        "jacc" : loss_jacc,
        "iou" : loss_iou
        }

    assert args.loss in loss_functions, "invalid loss function type"
    return loss_functions[args.loss]

import segnet
import unet

import tools.loss as loss
import tools.model_io as model_io

import torch.nn.functional as F
from torch.autograd import Variable


models = {
    "segnet" : segnet,
    "unet"   : unet
    }

def without(d, key):
    new_d = d.copy()
    new_d.pop(key)
    return new_d

def create(params):
    model_type = params['model']

    assert model_type in models, "invalid model type"
    return models[model_type].segmenter(**params['model_params'])


def save(path, model, model_params, epoch):
    state = {
        'epoch': epoch,
        'params': model_params,
        'state': model.state_dict()
    }

    model_io.save(path, epoch, state)

def load(path):

    state = model_io.load(path)
    params = state['params']
    model = create(params)

    model.load_state_dict(state['state'])

    return model, params, state['epoch']


def make_loss(args, num_classes):

    def loss_nll(output, labels):
        output = F.log_softmax(output)
        target = Variable(labels.cuda() if args.cuda else labels)

        return F.nll_loss(output, target)


    def loss_dice(output, labels):
        target = loss.one_hot(labels, num_classes)
        target = Variable(target.cuda() if args.cuda else target)

        return loss.dice(output, target)

    def loss_both(output, labels):
        return loss_nll(output, labels) + loss_dice(output, labels)


    loss_functions = {
        "nll" : loss_nll,
        "dice"   : loss_dice,
        "both" : loss_both
        }

    assert args.loss in loss_functions, "invalid loss function type"
    return loss_functions[args.loss]

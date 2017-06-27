import torch
from torch.autograd import Variable


from tools import Struct, confusion
import tools.tensor as tensor


def flatten_targets(probs, target, weights=None):
    assert probs.size() == target.size(), "Targets must have equal size to inputs"

    classes = probs.size(1)
    batch = probs.size(0)

    probs = probs.view(batch, classes, -1)
    target = target.view(batch, classes, -1).float()

    if weights:
        weights = weights.view(batch, 1, -1).expand_as(target)
    else:
        weights = 1

    return probs, target, weights

def weight(w, t):
    return w * t if w else t

def dice(probs, target, weights=None, size_average=False, class_average=False):
    p, t, w = flatten_targets(probs, target, weights)

    pt = (w * p * t).sum(2)
    p2 = (w * p * p).sum(2) + 1e-6
    t2 = (w * t * t).sum(2)

    dice = pt / (p2 + t2)
    n = dice.size(0) * dice.size(1)
    return n - dice.sum() # /(dice.size(0) * dice.size(1)) # average over batch and class


def iou(probs,target, weights=None):
    p, t, w = flatten_targets(probs, target, weights)

    pt = p * t
    d = p + t - pt

    iou = (w * pt).sum(2) / (w * d).sum(2)
    return 1 - iou.sum()/(iou.size(0) * iou.size(1)) # average over batch and class


def jacc(probs, target, weights=None):
    p, t, w = flatten_targets(probs, target, weights)

    pt = weight(w, p * t).sum(2)
    p2 = weight(w, p * p).sum(2) + 1e-6
    t2 = weight(w, t * t).sum(2)

    jacc = pt / (p2 + t2 - pt)
    return 1 - jacc.sum()/(jacc.size(0) * jacc.size(1)) # average over batch and class





def make_loss(name, num_classes, cuda=True):

    def var(labels):
        return Variable(labels.cuda() if cuda else labels)

    def loss_nll(output, labels, weights):
        return F.nll_loss(F.log_softmax(output), var(labels))

    def loss_dice(output, labels, weights=None):
        target = tensor.one_hot(labels, num_classes)
        return dice(output, var(target), var(weights) if weights else None)

    def loss_jacc(output, labels, weights=None):
        target = tensor.one_hot(labels, num_classes)
        return jacc(output, var(target), var(weights) if weights else None)

    def loss_iou(output, labels, weights=None):
        target = tensor.one_hot(labels, num_classes)
        return iou(output, var(target), var(weights) if weights else None)


    loss_functions = {
        "nll" : loss_nll,
        "dice"   : loss_dice,
        "jacc" : loss_jacc,
        "iou" : loss_iou
        }

    assert name in loss_functions, "invalid loss function type"
    return loss_functions[name]

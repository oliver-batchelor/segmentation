import torch
import torch.nn.functional as F
from torch.autograd import Variable


from tools import Struct, confusion
import tools.tensor as tensor


def flatten_targets(probs, target, weights=None):

    assert probs.size() == target.size(), "Targets must have equal size to inputs"

    classes = probs.size(1)
    batch = probs.size(0)

    probs = probs.view(batch, classes, -1).narrow(1, 1, classes - 1)
    target = target.view(batch, classes, -1).float().narrow(1, 1, classes - 1)

    if weights is not None:
        weights = weights.view(batch, 1, -1).expand_as(target)
    else:
        weights = 1

    return probs, target, weights


def dice(probs, target, weights=1, size_average=False, class_average=False):
    p, t, w = flatten_targets(probs, target, weights)

    pt = (w * p * t).sum(2)
    p2 = (w * p * p).sum(2) + 1e-6
    t2 = (w * t * t).sum(2)

    dice = pt / (p2 + t2)
    return 1 - 2 * dice.sum() / (dice.size(0) * dice.size(1)) # average over batch and class


def iou(probs,target, weights=1):
    p, t, w = flatten_targets(probs, target, weights)

    pt = (w * p * t).sum(2)
    p = (w * p).sum(2) + 1e-6
    t = (w * t).sum(2)

    iou = pt / (p + t - pt)
    return 1 - iou.sum()/(iou.size(0) * iou.size(1)) # average over batch and class

# def iou(probs,target, weights=None):
#     p, t, w = flatten_targets(probs, target, weights)
#
#     pt = p * t
#     d = p + t - pt + 1e-6
#
#     iou = (w * pt).sum(2) / (w * d).sum(2)
#     return 1 - iou.sum()/(iou.size(0) * iou.size(1)) # average over batch and class

def jacc(probs, target, weights=1):
    p, t, w = flatten_targets(probs, target, weights)

    pt = (w * p * t).sum(2)
    p2 = (w * p * p).sum(2) + 1e-6
    t2 = (w * t * t).sum(2)

    jacc = pt / (p2 + t2 - pt)
    return 1 - jacc.sum()/(jacc.size(0) * jacc.size(1)) # average over batch and class



def make_loss(name, num_classes, cuda=True):

    def var(labels):
        return Variable(labels.cuda() if cuda else labels)

    def loss_nll(output, labels, weights):
        return F.nll_loss(F.log_softmax(output), var(labels))

    def loss_dice(output, labels, weights):
        target = tensor.one_hot(labels, num_classes)
        return dice(F.softmax(output), var(target), var(weights))

    def loss_jacc(output, labels, weights):
        target = tensor.one_hot(labels, num_classes)
        return jacc(F.softmax(output), var(target), var(weights))

    def loss_iou(output, labels, weights):
        target = tensor.one_hot(labels, num_classes)
        return iou(F.softmax(output), var(target), var(weights))

    def loss_jacc_nll(outputs, labels, weights):
        return loss_jacc(outputs, labels, weights) + loss_nll(outputs, labels, weights)



    loss_functions = {
        "nll" : loss_nll,
        "dice"   : loss_dice,
        "jacc" : loss_jacc,
        "iou" : loss_iou,
        "jacc_nll" : loss_jacc_nll,
        }

    assert name in loss_functions, "invalid loss function type"
    return loss_functions[name]

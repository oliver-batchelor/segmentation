
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from tools import tensor


def one_hot(labels, num_classes):
    t = labels.new(labels.size(0), num_classes + 1).zero_()
    t.scatter_(1, labels.unsqueeze(1), 1)

    return t[:, 1:]

def all_eq(xs):
    return all(map(lambda x: x == xs[0], xs))




def focal_loss_softmax(class_target, class_pred, gamma=2, alpha=0.25, eps = 1e-6):
    #ce = F.cross_entropy(class_pred, class_target, size_average = False)

    p = F.softmax(class_pred, 1).clamp(eps, 1 - eps)
    p = p.gather(1, class_target.unsqueeze(1))

    errs = -(1 - p).pow(gamma) * p.log()

    return errs.sum()


def focal_loss_bce(class_target, class_pred, gamma=2, alpha=0.25, eps=1e-6):

    num_classes = class_pred.size(1)
    y = one_hot(class_target.detach(), num_classes).float()
    y_inv = 1 - y

    p_t = y * class_pred + y_inv * (1 - class_pred)
    a_t = y * alpha      + y_inv * (1 - alpha)

    p_t = p_t.clamp(min=eps, max=1-eps)

    errs = -a_t * (1 - p_t).pow(gamma) * p_t.log()
    return errs.sum()


def unpack(targets, predictions):
    loc_target, class_target =  targets
    loc_pred, class_pred = predictions

    size_of = lambda t: (t.size(0), t.size(1))
    sizes = list(map(size_of, [loc_target, class_target, loc_pred, class_pred]))
    assert all_eq (sizes), "total_loss: number of targets and predictions differ"

    pos_mask = (class_target > 0).unsqueeze(2).expand_as(loc_pred)
    loc_pred, loc_target = loc_pred[pos_mask], loc_target[pos_mask]

    valid_mask = class_target >= 0

    num_classes = class_pred.size(2)
    class_target = class_target[valid_mask]

    valid_mask = valid_mask.unsqueeze(2).expand_as(class_pred)
    class_pred = class_pred[valid_mask].view(-1, num_classes)

    return class_target, class_pred, loc_target, loc_pred, pos_mask.detach().cpu().sum().item()


def total_bce(targets, predictions, balance=5, gamma=2, alpha=0.25, eps=1e-6):
    class_target, class_pred, loc_target, loc_pred, n = unpack(targets, predictions)

    class_loss = focal_loss_bce(class_target, class_pred, gamma=gamma, alpha=alpha)
    loc_loss = F.smooth_l1_loss(loc_pred, loc_target, size_average=False)

    return class_loss / (n + 1.0), loc_loss * balance / (n + 1.0), n




    #return focal_loss(class_target, class_pred) + loc_loss(loc_target, loc_pred, class_target)


import torch
import torch.nn.functional as F
from torch.autograd import Variable

from tools import tensor


def one_hot(labels, num_classes):
    t = torch.cuda.FloatTensor(labels.size(0), num_classes).zero_()
    t.scatter_(1, labels.data.unsqueeze(1), 1)

    return Variable(t)

def focal_loss(class_target, class_pred, power = 2, eps = 1e-7):
    ce = F.cross_entropy(class_pred, class_target, size_average = False)
    p = F.softmax(class_pred, 1).clamp(eps, 1 - eps)
    p = p.gather(1, class_target.unsqueeze(1))

    losses = (1 - p).pow(power) * -p.log()
    return losses.sum()


def loss(loc_target, class_target, loc_pred, class_pred, balance = 1, power = 2):

    pos_mask = (class_target > 0).unsqueeze(2).expand_as(loc_pred)
    valid_mask = class_target >= 0

    num_classes = class_pred.size(2)

    class_target = class_target[valid_mask]
    class_pred = class_pred[valid_mask.unsqueeze(2).expand_as(class_pred)].view(-1, num_classes)

    loc_pred, loc_target = loc_pred[pos_mask], loc_target[pos_mask]
    num_positive = loc_pred.size(0)

    loss_class = focal_loss(class_target, class_pred, power = power)
    loss_box = balance * F.smooth_l1_loss(loc_pred, loc_target, size_average=False)

    return (loss_class + loss_box) / num_positive if loc_pred.dim() > 0 else 0

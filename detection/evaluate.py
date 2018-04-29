
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from detection import box

def cat(*xs, dim=0):
    def to_tensor(xs):
        return xs if torch.is_tensor(xs) else torch.FloatTensor([xs])
    return torch.cat([to_tensor(x) for x in xs], dim)

def rev_cummax(v):
    for i in range(v.size(0) - 1, 0, -1):
        v[i - 1] = max(v[i - 1], v[i])

    return v

def area_under_curve(xs, ys):
    i = (xs[1:] != xs[:-1]).nonzero().squeeze(1)

    return ((xs[i + 1] - xs[i]) * ys[i + 1]).sum()



def mAP(boxes_pred, labels_pred, confidence, boxes_truth, labels_truth, threshold=0.5, eps=1e-7):

    n = boxes_pred.size(0)
    m = boxes_truth.size(0)
    assert labels_pred.size(0) == n and confidence.size(0) == n

    ious = box.iou(boxes_pred, boxes_truth)

    true_positives = torch.FloatTensor(n).zero_()
    false_positives = torch.FloatTensor(n).zero_()

    for i in range(0, n):
        iou, j = ious[i].max(0)
        label = labels_truth[j[0]]

        if iou[0] > threshold:
            ious[:, j] = 0  # mark truth overlaps to 0 so they won't be selected twice

            if labels_pred[i] == label:
                true_positives[i] = 1
            else:
                false_positives[i] = 1

    true_positives = true_positives.cumsum(0)
    false_positives = false_positives.cumsum(0)

    recall = true_positives / m
    precision = true_positives / (true_positives + false_positives).clamp(min = eps)

    recall = cat(0.0, recall, 1.0)
    precision = rev_cummax(cat(1.0, precision, 0.0))

    return recall, precision, area_under_curve(recall, precision)

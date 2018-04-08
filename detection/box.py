# Originally by Alexander (Max) deGroot
# https://github.com/amdegroot/ssd.pytorch.git


import torch
from enum import Enum

def split(boxes):
    return boxes[..., :2],  boxes[..., 2:]

def split4(boxes):
    return boxes[..., 0],  boxes[..., 1], boxes[..., 2], boxes[..., 3]

def extents_form(boxes):
    lower, upper = split(boxes)
    return torch.cat([(lower + upper) * 0.5, upper - lower], 1)

def point_form(boxes):
    centre, size = split(boxes)
    extents = size * 0.5
    return torch.cat([centre - extents, centre + extents], 1)



def transform(boxes, offset, scale):
    lower, upper = boxes[:, :2], boxes[:, 2:]

    offset, scale = torch.Tensor(offset), torch.Tensor(scale)

    lower = lower.add(offset).mul(scale)
    upper = upper.add(offset).mul(scale)

    return torch.cat([lower.min(upper), lower.max(upper)], 1)


def filter_invalid(boxes, labels):
    valid = (boxes[:, 2] - boxes[:, 0] > 0) & (boxes[:, 3] - boxes[:, 1] > 0)
    inds = valid.nonzero()

    if inds.dim() > 1:
        inds = inds.squeeze(1)
        return torch.index_select(boxes, 0, inds), torch.index_select(labels, 0, inds)

    return torch.Tensor(), torch.LongTensor()


def clamp(boxes, lower, upper):

    boxes[:, 0].clamp_(min = lower[0])
    boxes[:, 1].clamp_(min = lower[1])
    boxes[:, 2].clamp_(max = upper[0])
    boxes[:, 3].clamp_(max = upper[1])

    return boxes


def intersect(box_a, box_b):
    """ Intersection of bounding boxes
    Args:
      box_a: (tensor) bounding boxes, Shape: [n,4].
      box_b: (tensor) bounding boxes, Shape: [m,4].
    Return:
      (tensor) intersection area, Shape: [n,m].
    """
    n = box_a.size(0)
    m = box_b.size(0)


    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(n, m, 2),
                       box_b[:, 2:].unsqueeze(0).expand(n, m, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(n, m, 2),
                       box_b[:, :2].unsqueeze(0).expand(n, m, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def iou(box_a, box_b):
    """Compute the IOU of two sets of boxes in point form.
    Args:
        box_a, box b: Bounding boxes in point form. shapes ([n, 4], [m, 4])
    Return:
        jaccard overlap: (tensor) Shape: [n, m]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [n,m]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [n,m]
    union = area_a + area_b - inter
    return inter / union  # [n,m]



def nms(bboxes, scores, threshold=0.5):
    '''Non maximum suppression.
    Args:
      bboxes: (tensor) bounding boxes in point form, sized [n,4].
      scores: (tensor) bbox scores, sized [n,].
      threshold: (float) overlap threshold.
    Returns:
      keep: (tensor) selected indices.
    Reference:
      https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py
    '''
    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]

    areas = (x2-x1) * (y2-y1)
    _, order = scores.sort(0, descending=True)

    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)

        if order.numel() == 1:
            break

        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])

        w = (xx2-xx1).clamp(min=0)
        h = (yy2-yy1).clamp(min=0)
        inter = w*h
        ovr = inter / areas[order[1:]].clamp(max=areas[i])
        ids = (ovr<=threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids+1]
    return torch.LongTensor(keep)


def match(boxes_target, class_target, boxes_pred, classes_pred, confs):
    overlaps = iou(boxes_target, boxes_pred)

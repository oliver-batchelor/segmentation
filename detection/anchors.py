import torch
import math

from detection import box



def make_boxes(box_sizes, box_dim, image_dim):
    w, h = box_dim

    n = len(box_sizes)

    xs = torch.arange(0, w).add_(0.5).view(1, w, 1, 1).expand(h, w, n, 1)
    ys = torch.arange(0, h).add_(0.5).view(h, 1, 1, 1).expand(h, w, n, 1)

    xs = xs.mul(image_dim[0] / w)
    ys = ys.mul(image_dim[1] / h)

    box_sizes = torch.Tensor(box_sizes).view(1, 1, n, 2).expand(h, w, n, 2)
    boxes = torch.cat([xs, ys, box_sizes], 3).view(-1, 4)

    return boxes


def make_anchors(box_sizes, layer_dims, image_dim, clamp=True):
    boxes = [make_boxes(boxes, box_dim, image_dim) for boxes, box_dim in zip(box_sizes, layer_dims)]
    boxes = torch.cat(boxes, 0)

    if clamp:
        return box.extents_form(box.clamp(box.point_form(boxes), (0, 0), image_dim))

    return boxes

def anchor_sizes(size, aspects, scales):
    def anchor(s, ar):
        return (s * math.sqrt(ar), s / math.sqrt(ar))

    return [anchor(size * scale, ar) for scale in scales for ar in aspects]


def encode(boxes, labels, anchor_boxes):
    '''Encode target bounding boxes and class labels.
    We obey the Faster RCNN box coder:
      tx = (x - anchor_x) / anchor_w
      ty = (y - anchor_y) / anchor_h
      tw = log(w / anchor_w)
      th = log(h / anchor_h)
    Args:
      boxes: (tensor) ground truth bounding boxes in point form, sized [n, 4].
      labels: (tensor) object class labels, sized [n].
      anchor_boxes: (tensor) bounding boxes in extents form, sized [m, 4].
    Returns:
      loc_targets: (tensor) encoded bounding boxes, sized [m, 4].
      class_targets: (tensor) encoded class labels, sized [m].
    '''
    assert tuple(boxes.size()) == (labels.size(0), 4)

    ious = box.iou(box.point_form(anchor_boxes), boxes)
    max_ious, max_ids = ious.max(1)

    ious = box.iou(box.point_form(anchor_boxes), boxes)
    boxes = boxes[max_ids]

    loc_xy = (boxes[:,:2]-anchor_boxes[:,:2]) / anchor_boxes[:,2:]
    loc_wh = torch.log(boxes[:,2:]/anchor_boxes[:,2:])
    loc_targets = torch.cat([loc_xy,loc_wh], 1)
    class_targets = 1 + labels[max_ids]

    class_targets[max_ious < 0.5] = 0 # negative label is 0
    ignore = (max_ious > 0.4) & (max_ious < 0.5)  # ignore ious between [0.4,0.5]
    class_targets[ignore] = -1  # mark ignored to -1

    return loc_targets, class_targets



def decode(loc_preds, class_preds, anchor_boxes, nms_threshold=0.3, class_threshold=0.05):
    '''Decode (encoded) predictions and anchor boxes to give detected boxes.
    Args:
      loc_preds: (tensor) box predictions in encoded form, sized [n, 4].
      class_preds: (tensor) object class predictions, sized [n].
      anchor_boxes: (tensor) bounding boxes in extents form, sized [m, 4].
    Returns:
      boxes: (tensor) detected boxes in point form, sized [k, 4].
      labels: (tensor) detected class labels [k].
    '''

    loc_pos, loc_sizes = box.split(loc_preds)
    anchor_pos, anchor_sizes = box.split(anchor_boxes)

    pos = loc_pos * anchor_sizes + anchor_pos
    sizes = loc_sizes.exp() * anchor_sizes

    boxes = box.point_form(torch.cat([pos, sizes], 1))

    score, labels = class_preds.max(1)
    ids = (score > class_threshold) & (labels > 0)
    ids = ids.nonzero().squeeze()
    keep = box.nms(boxes[ids], score[ids], threshold=nms_threshold).type_as(ids)

    return boxes[ids][keep], labels[ids][keep], score[ids][keep]

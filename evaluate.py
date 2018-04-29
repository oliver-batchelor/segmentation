import torch
import math

import torch.nn.functional as F
from torch.autograd import Variable

from tools.image import index_map
import tools.image.cv as cv

import tools.confusion as c

from tools.image.transforms import normalize_batch
from tools import Struct, tensor

from detection import evaluate

def eval_train(loss_func, var):

    def f(model, data):
        image, targets, lengths = data['image'], data['targets'], data['lengths']

        norm_data = var(normalize_batch(image))
        predictions = model(norm_data)

        class_loss, loc_loss, n = loss_func(var(targets), predictions)
        error = class_loss + loc_loss

        stats = Struct(error=error.data[0], class_loss=class_loss.data[0], loc_loss=loc_loss.data[0], size=image.size(0), boxes=lengths.sum(), matches=n)
        return Struct(error=error, statistics=stats)

    return f

def summarize_train(name, stats, epoch, globals={}):
    avg_loss = stats.error / stats.size
    avg_loc = stats.loc_loss / stats.size
    avg_class = stats.class_loss / stats.size
    avg_matched = stats.matches / stats.size
    avg_boxes= stats.boxes / stats.size

    print(name + ' epoch: {}\tBoxes (truth, matches) {:.2f} {:.2f} \tLoss (class, loc, total): {:.6f}, {:.6f}, {:.6f}'.format(epoch, avg_boxes, avg_matched, avg_class, avg_loc, avg_loss))
    return avg_loss


def eval_test(encoder, var):

    def f(model, data):

        images, target_boxes, target_labels = data['image'], data['boxes'], data['labels']
        assert images.size(0) == 1, "eval_test: expected batch size of 1 for evaluation"

        norm_data = var(normalize_batch(images))
        loc_preds, class_preds = model(norm_data)

        boxes, labels, confs = encoder.decode_batch(images, loc_preds.data, class_preds.data)[0]

        thresholds = [0.5 + inc * 0.05 for inc in range(0, 10)]
        scores = torch.FloatTensor(10).zero_()

        def mAP(iou):
            _, _, score = evaluate.mAP(boxes, labels, confs, target_boxes.type_as(boxes).squeeze(0), target_labels.type_as(labels).squeeze(0), threshold = iou)
            return score

        if(boxes.dim() > 0):
            scores = torch.FloatTensor([mAP(t) for t in thresholds])


        stats = Struct(AP=scores.mean(), mAPs=scores, size=1)
        return Struct(statistics=stats)

    return f

def summarize_test(name, stats, epoch, globals={}):
    mAPs =' '.join(['{:.2f}'.format(mAP * 100.0) for mAP in stats.mAPs / stats.size])
    AP = stats.AP * 100.0 / stats.size
    print(name + ' epoch: {}\t AP: {:.2f}\t mAPs@[0.5-0.95]: [{}]'.format(epoch, AP, mAPs))

    return AP

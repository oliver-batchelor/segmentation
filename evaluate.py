import torch
import math

import torch.nn.functional as F
from torch.autograd import Variable

from tools.image import index_map
import tools.image.cv as cv

import tools.confusion as c

from tools import Struct, tensor
from segmentation import transforms

import gc



def softmax(output):
    _, inds = output.max(1)
    return inds.long().squeeze(1).cpu()




def visualize(model, input, use_cuda=False, name="model", format="svg"):
    def to_cuda(t):
        return t.cuda() if use_cuda else t

    norm_data = transforms.normalize(input)
    vis.register_vis_hooks(model)

    output = model(Variable(to_cuda(norm_data)))

    vis.remove_vis_hooks()
    vis.save_visualization(name, format)


def statistics(confusion):
    num_labelled = confusion.float().sum(1)
    num_classified = confusion.float().sum(0)
    correct = confusion.float().diag()


    return Struct(
        precision = correct / num_classified.clamp(min=1),
        recall = correct / num_labelled.clamp(min=1),

        # -correct because the intersection is counted twice in num_classified + num_labelled
        iou = correct / (num_classified + num_labelled - correct).clamp(min=1),
        correct = correct.sum().item() / confusion.sum().item()
    )

def module(model, loss_func, classes, use_cuda=False, show=False, log=None):

    def to_cuda(t):
        return t.cuda() if use_cuda else t

    def run(data):
        image, labels, weights = data['image'], data['target'], data['weight']

        norm_data = transforms.normalize(to_cuda(image))
        output = model(Variable(norm_data))

        labels = tensor.centre_crop(labels, output.size())
        weights = tensor.centre_crop(weights, output.size())


        norm_labels, ignore_weight = transforms.normalize_target(to_cuda(labels), len(classes))
        weights = to_cuda(weights) * ignore_weight

        error = loss_func(output, norm_labels, weights)
        inds = softmax(output.data)

        confusion = c.confusion_matrix(inds, labels, len(classes))
    #    image_stats = sum([statistics(c.confusion_matrix(inds[i], labels[i], len(classes))) for i in range(0, image.size(0))])

        stats = Struct(error=error.item() * image.size(0), size=image.size(0), confusion=confusion)

        if show:
            overlay = index_map.overlay_batches(image, inds)
            if cv.display(overlay) == 27:
                exit(1)

        return Struct(error=error, output=output, prediction=inds, statistics=stats)

    def score(stats):
        return statistics(stats.confusion).iou.mean()

    def summarize(name, stats, epoch, globals={}):

        avg_loss = stats.error / stats.size
        # log.scalar(name + "/loss", avg_loss, step=epoch)
        #
        # for k, v in globals.items():
        #     log.scalar(name + "/" + k, v, step=epoch)

        print(name + ' epoch: {}\tLoss: {:.6f}'.format(epoch, avg_loss))
        if(len(classes) < 10):
            print(stats.confusion / stats.size)

        # image = stats.image / stats.size
        image = statistics(stats.confusion)
        precision, recall, iou = image.precision, image.recall, image.iou

        n = len(classes)

        print('{:16s} \t {} \t {} \t {}'.format("name:", "precision:", "recall:", "iou:"))
        for i in range(0, n):
            print('{:16s} \t {:10.2f} \t {:10.2f} \t {:10.2f}'.format(classes[i], precision[i] * 100, recall[i] * 100, iou[i] * 100))

        print('avg precision {:10.2f} '.format(precision.narrow(0, 1, n - 1).mean() * 100))
        print('avg recall    {:10.2f} '.format(recall.narrow(0, 1, n - 1).mean() * 100))

        # for i in range(1, n):
        #     prefix = name + "/classes/" + classes[i]
        #     # log.scalar(prefix + "/recall", recall[i], step=epoch)
        #     # log.scalar(prefix + "/precision", precision[i], step=epoch)
        #     log.scalar(prefix + "/iou", iou[i], step=epoch)

        total_correct = image.correct / stats.size
        # log.scalar(name + "/correct", total_correct, step=epoch)
        # log.flush()


    return Struct(summarize=summarize, run=run, score=score)

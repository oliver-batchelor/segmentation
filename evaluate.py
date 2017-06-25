import torch
import torch.nn.functional as F
from torch.autograd import Variable

from tools.image import index_map, transforms
import tools.image.cv as cv

import tools.model.loss as loss


import tools.logger as logger
from tools import Struct


def softmax(output):
    _, inds = F.softmax(output).data.max(1)
    return inds.long().squeeze(1).cpu()


class Statistics:
    def __init__(self, error, n, confusion):

        self.error = error
        self.confusion = confusion
        self.size = n

    def __add__(self, other):

        if other == 0:
            return self

        assert isinstance(other, Statistics)

        return Statistics(
            self.error     + other.error,
            self.size      + other.size,
            self.confusion + other.confusion)


    def __radd__(self, other):
        return self.__add__(other)




def module(model, loss_func, classes, use_cuda=False, show=False, log=logger.Null()):

    def to_cuda(t):
        return t.cuda() if use_cuda else t

    def run(data, labels, show=False):

        norm_data = transforms.normalize(to_cuda(data))
        norm_labels, weights = transforms.normalize_target(to_cuda(labels), len(classes))

        output = model(Variable(norm_data))
        #error = loss_func(output, norm_labels, weights)
        error = loss_func(output, norm_labels)
        inds = softmax(output)
        confusion = loss.confusion_matrix(inds, labels, len(classes))

        stats = Statistics(error, data.size(0), confusion)

        if show:
            overlay = index_map.overlay_batches(data, inds)
            if cv.display(overlay) == 27:
                exit(1)

        return Struct(error=error, output=output, prediction=inds, statistics=stats)



    def summarize(name, stats, epoch, globals={}):
        confusion = stats.confusion

        avg_loss = stats.error / stats.size
        log.scalar(name + "/loss", avg_loss, step=epoch)

        for k, v in globals.items():
            log.scalar(name + "/" + k, v, step=epoch)

        print(name + ' epoch: {}\tLoss: {:.6f}'.format(epoch, avg_loss))
        if(len(classes) < 10):
            print(confusion)

        num_labelled = confusion.float().sum(1).squeeze(1)
        num_classified = confusion.float().sum(0).squeeze(0)
        correct = confusion.float().diag()

        precision = correct / num_labelled
        recall = correct / num_classified

        n = len(classes)

        print('{:16s} \t {} \t {}'.format("name:", "precision:", "recall:"))
        for i in range(0, n):
            print('{:16s} \t {:10.2f} \t {:10.2f}'.format(classes[i], precision[i] * 100, recall[i] * 100))

        print('avg precision {:10.2f} '.format(precision.narrow(0, 1, n - 1).mean() * 100))
        print('avg recall    {:10.2f} '.format(recall.narrow(0, 1, n - 1).mean() * 100))

        for i in range(0, n):
            prefix = name + "/classes/" + classes[i]
            log.scalar(prefix + "/recall", recall[i], step=epoch)
            log.scalar(prefix + "/precision", precision[i], step=epoch)

        total_correct = correct.sum() / confusion.sum()
        log.scalar(name + "/correct", total_correct, step=epoch)


    return Struct(summarize=summarize, run=run, test=test)

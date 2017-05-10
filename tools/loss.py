import torch

def insert_size(s, dim, n):
    size = list(s)
    size.insert(dim, n)

    return torch.Size(size)

def one_hot(labels, classes, dim = 1):

    expanded = labels.view(insert_size(labels.size(), dim, 1))
    target = labels.new(insert_size(labels.size(), dim, classes))

    return target.zero_().scatter_(dim, expanded, 1)


def dice(probs,target):
    """
    probs is a torch variable of size BatchxnclassesxHxW representing log probabilities for each class
    target is a one-hot representation of the targets
    """

    assert probs.size() == target.size(), "Targets must have equal size to inputs"

    classes = probs.size(1)
    batch = probs.size(0)

    probs = probs.view(batch, classes, -1) #.narrow(1, 1, batch - 1)
    target = target.view(batch, classes, -1).float() #.narrow(1, 1, batch - 1)

    assert probs.size(2) == target.size(2), "targets and labels must have matching number of elements"

    num = probs.mul(target).sum(2)
    den1 = probs.mul(probs).sum(2)
    den2 = target.mul(target).sum(2)

    dice = 2*(num/(den1+den2))
    return 2 - dice.sum()/dice.size(0) #average over batch

def confusion_matrix(pred, target, num_classes = None):
    assert pred.size() == target.size(), "prediction must match target size"
    pred = pred.view(-1)
    target = target.view(-1)

    num_classes = num_classes or (max(pred.max(), target.max()) + 1)
    return count_elements(pred + target.mul(num_classes), num_classes * num_classes).view(num_classes, num_classes)


def confusion_zero(n):
    return torch.LongTensor (n, n).fill_(0)


def count_elements(indices, num_entries  = None):
    indices = indices.long().view(-1)

    num_entries = num_entries or (indices.max() + 1)
    c = torch.LongTensor(num_entries).fill_(0)

    ones = torch.LongTensor([1]).expand(indices.size(0))
    return c.index_add_(0, indices, ones)

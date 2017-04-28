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

    probs = probs.view(probs.size(0), probs.size(1), -1)
    target = target.view(target.size(0), target.size(1), -1).float()

    assert probs.size(2) == target.size(2), "Targets and labels must have matching number of elements"

    num = probs.mul(target).sum(2)
    den1 = probs.mul(probs).sum(2)
    den2 = target.mul(target).sum(2)

    dice = 2*(num/(den1+den2))
    return -1*dice.sum()/dice.size(0) #average over batch

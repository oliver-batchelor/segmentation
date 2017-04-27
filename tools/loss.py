import torch


def dice_loss(probs,target):
    """
    input is a torch variable of size BatchxnclassesxHxW representing log probabilities for each class
    target is a 1-hot representation of the groundtruth, shoud have same size as the input
    """
    assert input.size() == target.size(), "Input sizes must be equal."
    assert input.dim() == 4, "Input must be a 4D Tensor."
    uniques=np.unique(target.numpy())
    assert set(list(uniques))<=set([0,1]), "target must only contain zeros and ones"

    probs = probs.view(probs.size(0), probs.size(1), -1)
    target = probs.view(probs.size(0), probs.size(1), -1)

    num = probs.mul(target).sum(2)
    den1 = probs.mul(probs).sum(2)
    den2 = target.mul(target).sum(2)

    dice = 2*(num/(den1+den2)).select(1, 1)
    return -1*dice.sum()/dice.size(0) #average over batch


import torch

class RepeatSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly, repeating as necessary.
    Arguments:
        num_samples (int): number of samples per epoch
        max_element (int): maximum element index

    """

    def __init__(self, num_samples, dataset):
        self.num_samples = num_samples
        self.dataset = dataset

    def __iter__(self):
        return iter(torch.LongTensor(self.num_samples).random_(0, len(self.dataset) - 1))

    def __len__(self):
        return self.num_samples

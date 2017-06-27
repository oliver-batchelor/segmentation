import torch
import torch.nn as nn



def segmenter(encode, decode):

    class Pyramid(nn.Module):
        def __init__(self, inputs, depth):
            super().__init__()

            self.encode = encode(inputs, inputs * 2)
            self.inner = Pyramid(inputs * 2, depth - 1) if depth > 1 else None
            self.decode = decode(inputs * 2, inputs)

        def forward(self, input):

            output, inds = self.encode(input)
            if self.inner:
                output = self.inner(output)

            return self.decode(output, input, inds)


    class Segmenter(nn.Module):

        def __init__(self, num_classes = 2, input_channels = 3, features = 8, depth = 4):
            super().__init__()

            self.encode = encode(input_channels, features)
            self.decode = decode(features, num_classes)

            self.pyramid = Pyramid(features, depth)

        def forward(self, input):
            output, skip = self.encode(input)
            return self.decode(self.pyramid(output), skip)

    return Segmenter

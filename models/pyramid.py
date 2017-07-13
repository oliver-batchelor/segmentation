import torch
import torch.nn as nn



def segmenter(encode, decode, conv, growth=1.5):

    class Pyramid(nn.Module):
        def __init__(self, inputs, depth):
            super().__init__()

            outputs = int(inputs * growth)

            self.encode = encode(inputs, outputs)
            self.inner = Pyramid(outputs,  depth - 1) if depth > 0 else None
            self.decode = decode(outputs, self.encode.skip_size, inputs)

        def forward(self, input):

            output, skip = self.encode(input)
            if self.inner:
                output = self.inner(output)

            return self.decode(output, skip)


    class Segmenter(nn.Module):

        def __init__(self, input_channels=3, output_channels=2, features=8, depth=4):
            super().__init__()

            self.conv1 = conv(input_channels, features)
            self.conv2 = conv(features, output_channels)

            self.pyramid = Pyramid(features, depth)

        def forward(self, input):
            output = self.conv1(input)
            output = self.conv2(self.pyramid(output))
            return output

    return Segmenter

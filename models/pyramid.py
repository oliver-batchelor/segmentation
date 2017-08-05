import torch
import torch.nn as nn



def pyramid(encode, decode, growth=1.5):

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


    return Pyramid

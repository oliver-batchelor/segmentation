
import arguments
from dataset import detection, load
from detection.display import display_batch

from tools.image import cv

args = arguments.get_arguments()


classes, train, test = load.load_file(args.input)

def identity(batch):
    return batch

for i in train.train(args, collate_fn=identity):

    key = cv.display(display_batch(i, classes=classes))
    if(key == 27):
        break

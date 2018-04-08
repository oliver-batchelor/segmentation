
import arguments
from dataset.detection import annotate

from tools.image import cv

args = arguments.get_arguments()

with open(args.input, "r") as file:
    str = file.read()
    classes, train, test = annotate.load(str)

    for i in train.iter(args):

        key = cv.display(annotate.display_batch(i, classes=classes))
        if(key == 27):
            break

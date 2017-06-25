
from tools import cv, index_map

from torchvision import transforms
import argparse

import tools.transforms as transforms

parser = argparse.ArgumentParser(description='Image viewer for labelled images')
parser.add_argument('filename', help='Image file to view')


args = parser.parse_args()

image = cv.imread(args.filename).narrow(2, 0, 1).contiguous()


colorizer = index_map.colorizer(255)
cv.display(colorizer(image))

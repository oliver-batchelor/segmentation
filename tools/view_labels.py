
import conversions
from PIL import Image
import flat
from torchvision import transforms
import argparse



parser = argparse.ArgumentParser(description='Image viewer for labelled images')
parser.add_argument('filename', help='Image file to view')


args = parser.parse_args()

image = flat.load_target(args.filename)

labels = index_map.to_tensor(image)
colorizer = index_map.colorizer(255)

index_map.to_image(colorizer(labels)).show()

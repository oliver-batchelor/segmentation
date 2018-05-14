Make sure to checkout with --recurse-submodules, there's a submodule 'tools' for common bits and pieces)

It loads a folder full of images and masks (pixel value corresponds to class) with a config file.
It can be run - e.g. python main.py --lr 0.1 --batch_size 4 --input ~/storage/trees --model "unet --depth 5" --epoch_size 1024

An example dataset to train on can be found at:
https://drive.google.com/file/d/0B_mOCEqr7usZa2hra0xZUTJscE0/view

Or classes/images from the COCO/Pascal VOC dataset can be imported from scripts in the import/ folder.


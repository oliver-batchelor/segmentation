# Segmentation dataset

Make sure to checkout with --recurse-submodules, there's a submodule 'tools' for common bits and pieces)
It loads a folder full of images and masks (pixel value corresponds to class) with a config file.

An example dataset to train on can be found at:
https://drive.google.com/file/d/0B_mOCEqr7usZa2hra0xZUTJscE0/view

Or classes/images from the COCO/Pascal VOC dataset can be imported from scripts in the import/ folder.



# View the training or testing set and mask annotations:
  `python -m dataset.view --input /path/to/dataset --train (or --test)`

  Useful to check the preprocessing of images.

# View a mask file
  `python view_labels.py some/file.jpg.mask`

# Train a model:
  `python main.py --lr 0.1 --batch_size 4 --input /path/to/dataset --model "unet --depth 5" --epoch_size 1024`

## Common options:
  `--load`, load from a previous checkpoint and cointunue training

  `--model`, specify model and model parameters (use quotes)

  `--show`, show results of evaluating the model in training (sanity check)

# Evaluate a model on new image(s):
  `python test.py --batch /path/to/images --model log/model.pth --save results_path (and/or --show)`

  `python test.py --image /my/image.jpg --model log/model.pth --show`

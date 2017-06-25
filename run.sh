#~/bin/bash

rm -rf experiment
CMD="python main.py --lr 0.1 --depth 6 --batch_size 32 --epochs 128 --log experiment"

$CMD --model segnet  --nfeatures 16 --name segnet16_
$CMD --model unet  --nfeatures 16 --name unet16_
$CMD --model unet_full  --nfeatures 16 --name unet_full16_
$CMD --model segnet  --nfeatures 8 --name segnet8_
$CMD --model unet  --nfeatures 8 --name unet8_
$CMD --model unet_full  --nfeatures 8 --name unet_full8_

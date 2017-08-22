#~/bin/bash


COMMON="-m main --log log/coco --input /storage/workspace/coco_voc --lr 0.1 --batch_size 4 --image_size 440 --epoch_size 2048 --epochs 200 --max_scale=1.25 --min_scale=0.8 --rotation=5"
MODEL_COMMON="--base 32 --inc 32"



python $COMMON --name voc --model "pretrained --base_name resnet18 --block_layers 1  --dropout 0.0 --lr_modifier 0.1 $MODEL_COMMON" $@

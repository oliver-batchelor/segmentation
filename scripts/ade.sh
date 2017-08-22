#~/bin/bash

MODEL="pretrained --base 16 --inc 8 --block_layers 1"
python -m main --name ade_borders  --input /storage/workspace/ade_borders --lr 0.1 --max_scale=1.25 \
  --min_scale=0.8 --rotation=5 --loss jacc --batch_size 16 --model "$MODEL"  --epochs 50 --image_size 320  $@

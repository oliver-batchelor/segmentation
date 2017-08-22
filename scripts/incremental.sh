#~/bin/bash
TREES="--log log/depth/trees --input /storage/workspace/trees"
CATS="--log log/depth/cat --input /storage/workspace/voc_separate/cat"
PLANTS="--log log/depth/potted_plant --input /storage/workspace/voc_separate/potted_plant"
AEROPLANE="--log log/depth/aeroplane --input /storage/workspace/voc_separate/aeroplane"


PRETRAINED="pretrained --base_name resnet18 --lr_modifier 0.1 --base 16 --inc 8"
LADDER="ladder --base 16 --inc 0"


INC="-m main --lr 0.1 --batch_size 4 --image_size 440 --epoch_size 1024 --epochs 6"


for d in 1 2 3 4 5 6
  do
    python $INC $AEROPLANE --name $d-depth  --model "$PRETRAINED --depth $d"
    python $INC $CATS --name $d-depth  --model "$PRETRAINED --depth $d"
    python $INC $PLANTS --name $d-depth  --model "$PRETRAINED --depth $d"
    python $INC $TREES --name $d-depth  --model "$PRETRAINED --depth $d"

  done



# function run {
#     for n in $(eval echo "{1..$2}")
#     do
#       INC="-m main --limit $1 --lr 0.1 --batch_size 4 --image_size 440 --epoch_size 256 --epochs $3 --max_scale=1.25 --min_scale=0.8 --rotation=5 --seed $n"
#
#       python $INC $AEROPLANE --name "$1-limit" --model "$PRETRAINED"
#       python $INC $PLANTS --name "$1-limit" --model "$PRETRAINED"
#       python $INC $CATS  --name "$1-limit" --model "$PRETRAINED"
#       #python $INC $TREES --name "$1-limit" --model "$PRETRAINED"
#    done
# }
#
# # run 1 10 8
# # run 10 4 32
# # run 100 2 64
# # run 1000 1 128
# run 10000 1 128

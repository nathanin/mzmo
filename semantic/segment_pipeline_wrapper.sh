#!/bin/bash

# /home/nathan/semantic-ccrcc/code/segment_pipeline_wrapper.sh

set -e

proj="/home/nathan/mzmo"
# wgt="/50k/iter_50k.caffemodel"
weights="/home/nathan/semantic-pca/weights/seg_0.8/norm_resumed_iter_10000.caffemodel"
data="/home/nathan/data"
writedir="$proj/segmentation_0.8"

echo "Looking in $data"

# ls "$data/*ready"

python $proj/code/semantic/pipeline_segment_mzmo.py \
--path $proj/data/source_feature/256/0 \
--write_home $writedir/m0 \
--model $proj/code/semantic/segnet_basic_inference.prototxt \
--weights $weights

python $proj/code/semantic/pipeline_segment_mzmo.py \
--path $proj/data/source_feature/256/1 \
--write_home $writedir/m1 \
--model $proj/code/semantic/segnet_basic_inference.prototxt \
--weights $weights

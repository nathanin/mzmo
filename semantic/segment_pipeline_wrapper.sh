#!/bin/bash

# /home/nathan/semantic-ccrcc/code/segment_pipeline_wrapper.sh

set -e

proj="/home/nathan/mzmo"
# wgt="/50k/iter_50k.caffemodel"
wgt="/home/nathan/semantic-pca/weights/dec_7/norm_65000.caffemodel"
data="/home/nathan/data"
writedir="$proj/segmentation"
dropbox="~/Dropbox/projects/semantic_pca/test_dec21"

echo "Looking in $data"

# ls "$data/*ready"

python $proj/code/semantic/pipeline_segment_mzmo.py \
--path $proj/data/m0_ready \
--write_home $writedir/m0 \
--model $proj/code/semantic/segnet_basic_inference.prototxt \
--weights $wgt

python $proj/code/semantic/pipeline_segment_mzmo.py \
--path $proj/data/m1_ready \
--write_home $writedir/m1 \
--model $proj/code/semantic/segnet_basic_inference.prototxt \
--weights $wgt

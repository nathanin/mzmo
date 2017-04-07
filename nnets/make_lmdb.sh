#!/bin/bash

# /home/nathan/mzmo/code/make_lmdb.sh

set -e

caffe_tools="/home/nathan/caffe-segnet/build/tools"

# Training
echo "Training LMDB"
$caffe_tools/convert_imageset \
--resize_width 256 \
--resize_height 256 \
--shuffle \
/home/nathan/mzmo/data/training/ \
/home/nathan/mzmo/data/training/list.txt \
/home/nathan/mzmo/data/training/training_lmdb

echo "Training Meanfile"
$caffe_tools/compute_image_mean \
/home/nathan/mzmo/data/training/training_lmdb \
/home/nathan/mzmo/data/training/training_means.binaryproto



# Validation
echo "VALIDATION LMDB"
$caffe_tools/convert_imageset \
--resize_width 256 \
--resize_height 256 \
--shuffle \
/home/nathan/mzmo/data/validation/ \
/home/nathan/mzmo/data/validation/list.txt \
/home/nathan/mzmo/data/validation/validation_lmdb

echo "VALIDATION MEAN FILE"
$caffe_tools/compute_image_mean \
/home/nathan/mzmo/data/validation/validation_lmdb \
/home/nathan/mzmo/data/validation/validation_means.binaryproto



## Testing
#echo "Testing LMDB"
#$caffe_tools/convert_imageset \
#--resize_width 256 \
#--resize_height 256 \
#--shuffle \
#/home/nathan/mzmo/data/testing/ \
#/home/nathan/mzmo/data/testing/list.txt \
#/home/nathan/mzmo/data/testing/testing_lmdb
#
#
#echo "Testing Meanfile"
#$caffe_tools/compute_image_mean \
#/home/nathan/mzmo/data/testing/testing_lmdb \
#/home/nathan/mzmo/data/testing/testing_means.binaryproto

#!/bin/bash

#cf_root="/home/ubuntu/caffe-segnet/build/tools"
cf_root="/home/nathan/caffe-segnet/build/tools"

#proj="/home/ubuntu/mzmo"
proj="/home/nathan/mzmo"

# Do it
$cf_root/caffe train \
--solver $proj/code/solver_transfer.prototxt \
--weights $proj/weights/bvlc_alexnet.caffemodel
--log_dir $proj/weights/



#!/usr/bin/env sh

TOOLS=./build/tools

GLOG_logtostderr=0 GLOG_log_dir=models/90ksynth_v2/ \
  $TOOLS/caffe test \
  --model=models/90ksynth_v2/train_val.prototxt \
  --weights=models/90ksynth_v2/90ksynth_v2_v00_iter_100000.caffemodel \
  --iterations=3135 \
  --gpu=0

#  --weights=models/90ksynth_v2/90ksynth_v2_v00_iter_140000.caffemodel \

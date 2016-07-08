#!/usr/bin/env sh

TOOLS=./build/tools

GLOG_logtostderr=0 GLOG_log_dir=models/90ksynth_v2/ \
  $TOOLS/caffe train --solver=models/90ksynth_v2/solver.prototxt \
  --gpu=0,1


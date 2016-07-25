# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 16:22:00 2016

@author: marfsac
"""

# Test the solver for MNIST with the spatial transformer
import numpy as np
import matplotlib.pyplot as plt
import lmdb
from PIL import Image
from StringIO import StringIO
import sys
sys.path.append("/Users/marissac/caffe/examples/pycaffe/layers")

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Make sure that caffe is on the python path:
caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
import os
os.chdir(caffe_root)

import caffe

from google.protobuf import text_format
from caffe.proto import caffe_pb2

model_def = '/Users/marissac/caffe/examples/ssd/models/VGGNet/cocoText/SSD_300x300/test_annoIn.prototxt'
#model_weights = '/Users/marissac/caffe/examples/ssd/models/VGGNet/cocoText/SSD_300x300/VGG_cocoText_legibleSplit_SSD_300x300_iter_60000.caffemodel'


net = caffe.Net(model_def,      # defines the structure of the model
                caffe.TRAIN)     # use test mode (e.g., don't perform dropout)
                
net.forward()

test_label = net.blobs['label'].data[0,0,:,:]

lmdb_env = lmdb.open('/Users/marissac/data/coco/lmdb/coco_combo_legibleOnly_train2014_final_lmdb')
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()
datum = caffe.proto.caffe_pb2.AnnotatedDatum()
datum_anno_group = caffe.proto.caffe_pb2.AnnotationGroup()

for k in range(0,7):
    lmdb_cursor.next_nodup()
    value = lmdb_cursor.value()
    datum.ParseFromString(value)
    datum_anno_group.ParseFromString(value)
    jpg = datum.datum.data
    
    test_out = datum_anno_group.group_label
    print test_out

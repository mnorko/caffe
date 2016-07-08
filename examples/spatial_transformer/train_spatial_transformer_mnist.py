# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 11:02:27 2016

@author: marissac
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

model_def = '/Users/marissac/caffe/examples/spatial_transformer/solver.prototxt'
net_def = '/Users/marissac/caffe/examples/spatial_transformer/train_st_cnn.prototxt'
#solver = caffe.SGDSolver(model_def)
net = caffe.Net(net_def,caffe.TRAIN)

#solver.step(1)
net.forward()


net.backward()
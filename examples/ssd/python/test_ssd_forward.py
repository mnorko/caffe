# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 09:52:46 2016

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

model_def = '/Users/marissac/caffe/examples/ssd/models/VGGNet/cocoText/SSD_300x300/solver.prototxt'
model_weights = '/Users/marissac/caffe/examples/ssd/models/VGGNet/cocoText/SSD_300x300/VGG_cocoText_legibleSplit_SSD_300x300_iter_60000.caffemodel'

solver = caffe.SGDSolver(model_def)
solver.net.copy_from(model_weights)
#net = caffe.Net(model_def,model_weights,caffe.TRAIN)

#solver.step(1)
solver.step(1)


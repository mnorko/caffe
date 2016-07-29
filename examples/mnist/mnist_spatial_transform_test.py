# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 10:02:50 2016

@author: marissac
"""

import numpy as np
import caffe
import sys
sys.path.append("/Users/marissac/caffe/examples/pycaffe/layers")
import matplotlib.pyplot as plt

model_def = '/Users/marissac/caffe/examples/mnist/lenet_train_test_spatial_transform.prototxt'
model_weights = '/Users/marissac/caffe/examples/mnist/spatial_trans_test/ST_CNN_constrainedTheta_iter_80000.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)
                
net.forward()

imgNum = 0
digitIn = net.blobs['data'].data[imgNum,0,:,:]
digitTrans = net.blobs['st_output'].data[imgNum,0,:,:]
theta = net.blobs['theta'].data

plt.figure(0)
plt.imshow(digitIn)

plt.figure(1)
plt.imshow(digitTrans)
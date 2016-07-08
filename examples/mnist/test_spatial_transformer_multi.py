# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 16:55:08 2016

@author: marissac
"""

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

model_def = '/Users/marissac/caffe/examples/spatial_transformer/deploy_lenet_spatial_transform.prototxt'
model_weights = '/Users/marissac/caffe/examples/mnist/lenet_iter_10000.caffemodel'


net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)
                
#net_test = caffe.Net(model_test,caffe.TEST)
#net_test.forward()
                
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_raw_scale('data', 1/255)


lmdb_env = lmdb.open('/Users/marissac/caffe/examples/mnist/mnist_test_lmdb')
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()
datum = caffe.proto.caffe_pb2.Datum()

#for key, value in lmdb_cursor:
    
lmdb_cursor.next_nodup()
value = lmdb_cursor.value()
datum.ParseFromString(value)
label = datum.label
image = caffe.io.datum_to_array(datum)
test= 1

transformed_image = image/256.0
# Create an image with an MNIST digit had a specified location
width_input = 84.0
height_input = 84.0
imgTemp = np.zeros((height_input,width_input))
loc_digit = np.array([[10,20],[52,20]]) # box x location, box y location
num_detections = 2
for loc_idx in range(0,num_detections):
    imgTemp[loc_digit[loc_idx,1]:loc_digit[loc_idx,1]+28,loc_digit[loc_idx,0]:loc_digit[loc_idx,0]+28] =transformed_image[0,:,:]
#imgTemp = transformed_image[0,:,:]
imgUse = np.expand_dims(imgTemp,0)

# Find digit
bbox_size_x = 28.0
bbox_size_y = 28.0
scale_x = bbox_size_x/width_input
scale_y = bbox_size_y/height_input
theta = np.zeros((num_detections,6))
for loc_idx in range(0,num_detections):
    bbox_x = 2*(-(width_input-bbox_size_x)/2 + loc_digit[loc_idx,0])/width_input
    bbox_y = 2*(-(height_input-bbox_size_y)/2 + loc_digit[loc_idx,1])/height_input
    theta[loc_idx,:] = np.array([scale_x,0,bbox_x,0,scale_y,bbox_y])
#theta = np.array([1,0,0,0,1,0])


net.blobs['data'].data[...] = imgUse
net.blobs['theta'].data[...] = theta
lmdb_cursor.next_nodup()
value = lmdb_cursor.value()
datum.ParseFromString(value)
label = datum.label
data = caffe.io.datum_to_array(datum)
test= 1
    
net.forward()   
#transformed_image = transformer.preprocess('data',image)
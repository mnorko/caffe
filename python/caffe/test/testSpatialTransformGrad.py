# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 15:17:48 2016

@author: marissac
"""

import numpy as np

import caffe
from caffe.proto import caffe_pb2
import sys
sys.path.append("/Users/marissac/caffe/examples/pycaffe/layers")

def get_obj_and_gradient(top, top_id, top_data_id):
        for b in top:
            b.diff[...] = 0
        loss_weight = 2
        loss = top[top_id].data.flat[top_data_id] * loss_weight
        top[top_id].diff.flat[top_data_id] = loss_weight
        return loss
        
# Create a layer with random inputs

bottom = dict()
shape = [1,1,2,2]
theta_shape = [1,6]
seed=1701

img = caffe.Blob(shape)
theta = caffe.Blob(theta_shape)

#img.data[...] = np.random.rand(1,1,2,2)
img.data[0,0,:,:] = np.array([[0.1,0.2],[0.3,0.4]])
#img.data[0,0,:,:] = np.array([[0.1,0.2,0.3],[0.4,0.5,0.6],[0.7,0.8,0.9]])
#theta.data[...] = np.array([0.8,0.1,0,0.1,0.8,0])
theta.data[...] = np.array([0.99,0,0,0,0.99,0])
bottom[0] = img
bottom[1] = theta

top = [caffe.Blob([])]

# Specify Layer parameters
lp = caffe_pb2.LayerParameter()
lp.type = "Python"
lp.python_param.module = "spatialTransformer"
lp.python_param.layer = "SpatialTransformerLayer"
lp.python_param.param_str = "{'output_H': 2, 'output_W': 2}"
layer = caffe.create_layer(lp)
layer.SetUp(bottom,top)

propagate_down = [True, True]

caffe.set_random_seed(seed)
layer.Reshape(bottom, top)
layer.Forward(bottom, top)

loss_weight = 2
top_data_id = 2
top[0].diff[...] = loss_weight*np.zeros(top[0].data.shape)
top[0].diff.flat[top_data_id] = loss_weight
top_diff_vals = top[0].diff[0,0,:,:]
        
layer.Backward(top, propagate_down, bottom)
bottom_img_diff = bottom[0].diff[0,0,:,:]
bottom_theta_diff = bottom[1].diff

step = 1e-4
bottom_num = 1
fi = 4
bottom[bottom_num].data.flat[fi] += step
caffe.set_random_seed(seed)
layer.Reshape(bottom, top)
layer.Forward(bottom, top)
# L(fi <-- fi-step)
#top[0].diff[...] = loss_weight*np.zeros(top[0].data.shape)
#ploss = top[0].data.flat[top_data_id] * loss_weight
top_data_p = top[0].data
ploss = get_obj_and_gradient(top, 0, top_data_id)
top[0].diff.flat[top_data_id] = loss_weight

bottom[bottom_num].data.flat[fi] -= 2*step
caffe.set_random_seed(seed)
layer.Reshape(bottom, top)
layer.Forward(bottom, top)
#top[0].diff[...] = loss_weight*np.zeros(top[0].data.shape)
#nloss = top[0].data.flat[top_data_id] * loss_weight
top_data_n = top[0].data
nloss = get_obj_and_gradient(top, 0, top_data_id)

top[0].diff.flat[top_data_id] = loss_weight
grad = (ploss - nloss) / (2. * step)


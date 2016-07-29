# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 12:20:34 2016

@author: marissac
"""
import numpy as np

import caffe
from caffe.proto import caffe_pb2
import sys
sys.path.append("/Users/marissac/caffe/examples/pycaffe/layers")


def get_obj_and_gradient(top, top_id, top_data_id):
        for b in range(0,len(top)):
            top[b].diff[...] = 0
        loss_weight = 2
        loss = top[top_id].data.flat[top_data_id] * loss_weight
        top[top_id].diff.flat[top_data_id] = loss_weight
        return loss
        
pred_bbox = np.array([[0.2,0.2,0.6,0.6]])
pred_bbox = np.expand_dims(pred_bbox,axis = 2)
pred_bbox = np.expand_dims(pred_bbox,axis = 3)


gt_xmin = 0.24
gt_ymin = 0.24
gt_xmax = 0.6
gt_ymax = 0.6
gt_data = np.zeros((1,33))
gt_data[0,3:7] = np.array([gt_xmin,gt_ymin,gt_xmax,gt_ymax])
gt_data[0,8] = 30
gt_data[0,9] = 30
gt_data = np.expand_dims(gt_data,axis=2)
gt_data = np.expand_dims(gt_data,axis=3)

shape = [1,3,20,20]
rng = np.random.RandomState(313)
img = caffe.Blob(shape)
img.data[...] = rng.randn(*shape)


conf_data = np.array([0.4,0.6])
conf_data = np.expand_dims(conf_data,axis = 0)


pred_in = caffe.Blob(pred_bbox.shape)
gt_in = caffe.Blob(gt_data.shape)

bottom = dict()
top = dict()
pred_in.data[...] = pred_bbox
gt_in.data[...] = gt_data
bottom[0] =pred_in
bottom[1] = gt_in
bottom[2] = img
top[0] = caffe.Blob((1,6))
top[1] = caffe.Blob((1,1,20,20))
top[2] = caffe.Blob((1,23))

lp = caffe_pb2.LayerParameter()
lp.type = "Python"
lp.python_param.module = "bboxToTransformLayer"
lp.python_param.layer = "BboxToTransformLayer"
layer = caffe.create_layer(lp)
layer.SetUp(bottom, top)
layer.Reshape(bottom, top)
layer.Forward(bottom, top)

loss_weight = 2
top_data_id = 2
top_id = 0
top[top_id].diff[...] = loss_weight*np.zeros(top[top_id].data.shape)
top[top_id].diff.flat[top_data_id] = loss_weight

seed=1701  
propagate_down = [True, False, False]
layer.Backward(top, propagate_down, bottom)
bottom_pred = bottom[0].diff[0,:,0,0]

step = 1e-4
bottom_num = 0
fi = 0
bottom[bottom_num].data.flat[fi] += step
caffe.set_random_seed(seed)
layer.Reshape(bottom, top)
layer.Forward(bottom, top)
# L(fi <-- fi-step)
#top[0].diff[...] = loss_weight*np.zeros(top[0].data.shape)
#ploss = top[0].data.flat[top_data_id] * loss_weight
ploss = get_obj_and_gradient(top, top_id, top_data_id)
top[top_id].diff.flat[top_data_id] = loss_weight

bottom[bottom_num].data.flat[fi] -= 2*step
caffe.set_random_seed(seed)
layer.Reshape(bottom, top)
layer.Forward(bottom, top)
#top[0].diff[...] = loss_weight*np.zeros(top[0].data.shape)
#nloss = top[0].data.flat[top_data_id] * loss_weight
top_data_n = top[top_id].data
nloss = get_obj_and_gradient(top, top_id, top_data_id)

top[top_id].diff.flat[top_data_id] = loss_weight
grad = (ploss - nloss) / (2. * step)
agrad = bottom[bottom_num].diff.flat[fi]
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 11:11:59 2016

@author: marissac
"""
import numpy as np

import caffe
from caffe.proto import caffe_pb2
import sys
sys.path.append("/Users/marissac/caffe/examples/pycaffe/layers")
import multibox_util

def get_obj_and_gradient(top, top_id, top_data_id):
        for b in range(0,len(top)):
            top[b].diff[...] = 0
        loss_weight = 2
        loss = top[top_id].data.flat[top_data_id] * loss_weight
        top[top_id].diff.flat[top_data_id] = loss_weight
        return loss
        
loc_data_init = np.array([0.2,0.2,0.6,0.6])
prior_data = np.array([[0.21,0.21,0.59,0.59],[0.1,0.1,0.1,0.1]])  
gt_xmin = 0.24
gt_ymin = 0.24
gt_xmax = 0.6
gt_ymax = 0.6
gt_data = np.zeros((1,33))
gt_data[0,3:7] = np.array([gt_xmin,gt_ymin,gt_xmax,gt_ymax])
gt_data[0,8] = 100
gt_data[0,9] = 100
gt_data = np.expand_dims(gt_data,axis=0)
gt_data = np.expand_dims(gt_data,axis=0)
prior_bboxes = []
prior_bboxes.append({'xmin':prior_data[0,0],'ymin':prior_data[0,1],'xmax':prior_data[0,2],'ymax':prior_data[0,3]})
prior_var = []
prior_var.append({'xmin':prior_data[1,0],'ymin':prior_data[1,1],'xmax':prior_data[1,2],'ymax':prior_data[1,3]})

gt_bbox = []
gt_bbox.append({'label':0,'xmin':gt_xmin,'ymin':gt_ymin,'xmax':gt_xmax,'ymax':gt_ymax})

pred_bbox =[]
pred_bbox.append({'xmin':loc_data_init[0],'ymin':loc_data_init[1],'xmax':loc_data_init[2],'ymax':loc_data_init[3]})


loc_data = multibox_util.encodeBBoxes(prior_bboxes,prior_var,pred_bbox,np.array([0]),np.array([0]))
conf_data = np.array([0.4,0.6])
conf_data = np.expand_dims(conf_data,axis = 0)

prior_data = np.expand_dims(prior_data,axis = 0)

loc_in = caffe.Blob(loc_data.shape)
conf_in = caffe.Blob(conf_data.shape)
prior_in = caffe.Blob(prior_data.shape)
gt_in = caffe.Blob(gt_data.shape)

bottom = dict()
top = dict()
loc_in.data[...] = loc_data
conf_in.data[...] = conf_data
prior_in.data[...] = prior_data
gt_in.data[...] = gt_data
bottom[0] =loc_in
bottom[1] = conf_in
bottom[2] = prior_in
bottom[3] = gt_in
top[0] = caffe.Blob((1,4,1,1))
top[1] = caffe.Blob((1,2))
top[2] = caffe.Blob((1,4))
top[3] = caffe.Blob((1,33))
top[4] = caffe.Blob((1,1))
top[5] = caffe.Blob((1,4)) 

seed=1701        
lp = caffe_pb2.LayerParameter()
lp.type = "Python"
lp.python_param.module = "multiboxRoutingLayer"
lp.python_param.layer = "MultiboxRoutingLayer"
lp.python_param.param_str = "{'max_matches': 1,'overlap_threshold': 0.5}"
layer = caffe.create_layer(lp)
layer.SetUp(bottom,top)
propagate_down = [True, True, False, False]
caffe.set_random_seed(seed)
layer.Reshape(bottom, top)
layer.Forward(bottom, top)

loss_weight = 2
top_data_id = 0
top_id = 0
top[top_id].diff[...] = loss_weight*np.zeros(top[top_id].data.shape)
top[top_id].diff.flat[top_data_id] = loss_weight


layer.Backward(top, propagate_down, bottom)
bottom_loc = bottom[0].diff
bottom_conf = bottom[1].diff
bottom_prior = bottom[2].diff
bottom_labels = bottom[3].diff

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




# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 10:36:01 2016

@author: marissac
"""

import unittest
import numpy as np

import caffe
from caffe.proto import caffe_pb2
import sys
sys.path.append("/Users/marissac/caffe/examples/pycaffe/layers")
from caffe.gradient_check_util import GradientChecker
import multibox_util

class TestGradientChecker():

    def setUp(self):

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
        
        self.bottom = dict()
        self.top = dict()
        loc_in.data[...] = loc_data
        conf_in.data[...] = conf_data
        prior_in.data[...] = prior_data
        gt_in.data[...] = gt_data
        self.bottom[0] =loc_in
        self.bottom[1] = conf_in
        self.bottom[2] = prior_in
        self.bottom[3] = gt_in
        self.top[0] = caffe.Blob((1,4,1,1))
        self.top[1] = caffe.Blob((1,2))
        self.top[2] = caffe.Blob((1,4))
        self.top[3] = caffe.Blob((1,33))
        self.top[4] = caffe.Blob((1,1))
        self.top[5] = caffe.Blob((1,4)) 

    def test_euclidean(self):
           
        
        lp = caffe_pb2.LayerParameter()
        lp.type = "Python"
        lp.python_param.module = "multiboxRoutingLayer"
        lp.python_param.layer = "MultiboxRoutingLayer"
        lp.python_param.param_str = "{'max_matches': 1,'overlap_threshold': 0.5}"
        layer = caffe.create_layer(lp)
        layer.SetUp(self.bottom, self.top)
        layer.Reshape(self.bottom, self.top)
        layer.Forward(self.bottom, self.top)
        checker = GradientChecker(1e-5, 3e-2)
        checker.check_gradient_exhaustive(
            layer, self.bottom, self.top, check_bottom=[1])

            
testGrad = TestGradientChecker()
TestGradientChecker.setUp(testGrad)
TestGradientChecker.test_euclidean(testGrad)
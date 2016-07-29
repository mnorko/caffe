# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 11:59:46 2016

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

        
        pred_bbox = np.array([[0.2,0.2,0.6,0.6]])
        pred_bbox = np.expand_dims(pred_bbox,axis = 2)
        pred_bbox = np.expand_dims(pred_bbox,axis = 3)
        
        
        gt_xmin = 0.24
        gt_ymin = 0.24
        gt_xmax = 0.6
        gt_ymax = 0.6
        gt_data = np.zeros((1,33))
        gt_data[0,3:7] = np.array([gt_xmin,gt_ymin,gt_xmax,gt_ymax])
        gt_data[0,8] = 20
        gt_data[0,9] = 20
        gt_data = np.expand_dims(gt_data,axis=2)
        gt_data = np.expand_dims(gt_data,axis=3)
        
        shape = [1,3,20,20]
        self.rng = np.random.RandomState(313)
        img = caffe.Blob(shape)
        img.data[...] = self.rng.randn(*shape)
        
        
        conf_data = np.array([0.4,0.6])
        conf_data = np.expand_dims(conf_data,axis = 0)
        
        
        pred_in = caffe.Blob(pred_bbox.shape)
        gt_in = caffe.Blob(gt_data.shape)
        
        self.bottom = dict()
        self.top = dict()
        pred_in.data[...] = pred_bbox
        gt_in.data[...] = gt_data
        self.bottom[0] =pred_in
        self.bottom[1] = gt_in
        self.bottom[2] = img
        self.top[0] = caffe.Blob((1,6))
        self.top[1] = caffe.Blob((1,1,20,20))
        self.top[2] = caffe.Blob((1,23))


    def test_euclidean(self):
           
        
        lp = caffe_pb2.LayerParameter()
        lp.type = "Python"
        lp.python_param.module = "bboxToTransformLayer"
        lp.python_param.layer = "BboxToTransformLayer"
        layer = caffe.create_layer(lp)
        layer.SetUp(self.bottom, self.top)
        layer.Reshape(self.bottom, self.top)
        layer.Forward(self.bottom, self.top)
        checker = GradientChecker(1e-5, 3e-2)
        checker.check_gradient_exhaustive(
            layer, self.bottom, self.top, check_bottom=[0])

            
testGrad = TestGradientChecker()
TestGradientChecker.setUp(testGrad)
TestGradientChecker.test_euclidean(testGrad)
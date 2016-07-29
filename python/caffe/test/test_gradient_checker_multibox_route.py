# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 10:37:42 2016

@author: marissac
"""

import unittest
import numpy as np

import caffe
from caffe.proto import caffe_pb2
import sys
sys.path.append("/Users/marissac/caffe/examples/pycaffe/layers")
from caffe.gradient_check_util import GradientChecker

class TestGradientChecker():

    def setUp(self):
        #model_def = '/Users/marissac/caffe/examples/ssd/models/VGGNet/cocoText/SSD_300x300/test_annoWord.prototxt'
        #model_weights = '/Users/marissac/caffe/examples/ssd/models/VGGNet/cocoText/SSD_300x300/VGG_cocoText_legibleSplit_SSD_300x300_iter_60000.caffemodel'
        
        combo_weights = '/Users/marissac/caffe/examples/ocr/detect_2444000_read_140000_combo_final.caffemodel'
        model_combo = '/Users/marissac/caffe/examples/ocr/train_detectReadCombo.prototxt'
        net = caffe.Net(model_combo,combo_weights,caffe.TEST)
        net.forward()
        
        gt_data = net.blobs['label'].data

        prior_data = net.blobs['mbox_priorbox'].data
        loc_data = net.blobs['mbox_loc'].data
        conf_data = net.blobs['mbox_conf'].data
        
        
              
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
        self.top[0] = caffe.Blob((2,4,20,1))
        self.top[1] = caffe.Blob((40,83))
        self.top[2] = caffe.Blob((2,80,1,1))
        self.top[3] = caffe.Blob((40,33))
        self.top[4] = caffe.Blob((40,1))
        self.top[5] = caffe.Blob((2,80)) 

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
            layer, self.bottom, self.top, check_bottom=[0])

            
testGrad = TestGradientChecker()
TestGradientChecker.setUp(testGrad)
TestGradientChecker.test_euclidean(testGrad)

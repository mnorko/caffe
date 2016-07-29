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
        self.bottom = dict()
        shape = [1,1,20,20]
        pred = caffe.Blob(shape)
        theta_shape = [2,6]
        label = caffe.Blob(theta_shape)
        self.rng = np.random.RandomState(313)
        pred.data[...] = self.rng.randn(*shape)
        label.data[...] = np.array([[0.99,0,0,0,0.99,0],[0.99,0,0,0,0.99,0]])

        self.bottom[0] =pred
        self.bottom[1] = label
        self.top = [caffe.Blob([])]

    def test_euclidean(self):
        lp = caffe_pb2.LayerParameter()
        lp.type = "Python"
        lp.python_param.module = "spatialTransformerMulti"
        lp.python_param.layer = "SpatialTransformerMultiLayer"
        lp.python_param.param_str = "{'output_H': 2, 'output_W': 4, 'num_detections': 2}"
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

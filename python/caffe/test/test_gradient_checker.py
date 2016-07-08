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
        shape = [1,1,4,4]
        pred = caffe.Blob(shape)
        theta_shape = [2,6]
        label = caffe.Blob(theta_shape)
        self.rng = np.random.RandomState(313)
        pred.data[...] = self.rng.randn(*shape)
        label.data[...] = np.array([0.5,0.1,0.1,0.1,0.5,0.1])
        label.data[...] = np.array([[0.99,0,0,0,0.99,0],[0.99,0,0,0,0.99,0]])
        print label.data
        self.bottom[0] =pred
        self.bottom[1] = label
        self.top = [caffe.Blob([])]
        #print pred.shape
        #print label.shape

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
        # manual computation
        #loss = np.sum((self.bottom[0].data - self.bottom[1].data) ** 2) \
        #    / self.bottom[0].shape[0] / 2.0
        #self.assertAlmostEqual(float(self.top[0].data), loss, 5)
        checker = GradientChecker(1e-5, 3e-2)
        checker.check_gradient_exhaustive(
            layer, self.bottom, self.top, check_bottom='all')

#    def test_inner_product(self):
#        lp = caffe_pb2.LayerParameter()
#        lp.type = "Python"
#        lp.python_param.module = "spatialTransformer"
#        lp.python_param.layer = "SpatialTransformerLayer"
#        lp.python_param.param_str = "{'output_H': 28, 'output_W': 28}"
#        layer = caffe.create_layer(lp)
#        layer.SetUp([self.bottom[0]], self.top)
#        w = self.rng.randn(*layer.blobs[0].shape)
#        b = self.rng.randn(*layer.blobs[1].shape)
#        layer.blobs[0].data[...] = w
#        layer.blobs[1].data[...] = b
#        layer.Reshape([self.bottom[0]], self.top)
#        layer.Forward([self.bottom[0]], self.top)
#        assert np.allclose(
#            self.top[0].data,
#            np.dot(
#                self.bottom[0].data.reshape(self.bottom[0].shape[0], -1), w.T
#                ) + b
#            )
#        checker = GradientChecker(1e-2, 1e-1)
#        checker.check_gradient_exhaustive(
#            layer, [self.bottom[0]], self.top, check_bottom=[0])
            
testGrad = TestGradientChecker()
TestGradientChecker.setUp(testGrad)
TestGradientChecker.test_euclidean(testGrad)
#if __name__ == '__main__':
#    unittest.main()

#bottom = dict() 
#shape = [1,1,28,28]
#theta_shape = [1,6]
#pred = caffe.Blob(shape)
#label = caffe.Blob(theta_shape)
#rng = np.random.RandomState(313)
#pred.data[...] = rng.randn(*shape)
#label.data[...] = rng.randn(*theta_shape)
#
#bottom[0] = pred
#bottom[1] = label
#top = [[caffe.Blob([])]]
#
#lp = caffe_pb2.LayerParameter()
#lp.type = "Python"
#lp.python_param.module = "spatialTransformer"
#lp.python_param.layer = "SpatialTransformerLayer"
#lp.python_param.param_str = "{'output_H': 28, 'output_W': 28}"
#layer = caffe.create_layer(lp)
#layer.setup(bottom, top)
#layer.reshape(bottom, top)
#layer.forward(bottom, top)
## manual computation
#loss = np.sum((bottom[0].data - bottom[1].data) ** 2) \
#    / bottom[0].shape[0] / 2.0
##assertAlmostEqual(float(top[0].data), loss, 5)
#checker = GradientChecker(1e-2, 1e-2)
#checker.check_gradient_exhaustive(
#    layer, bottom, top, check_bottom='all')
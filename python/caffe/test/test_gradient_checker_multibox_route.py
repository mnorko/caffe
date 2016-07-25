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
        model_def = '/Users/marissac/caffe/examples/ssd/models/VGGNet/cocoText/SSD_300x300/test_annoWord.prototxt'
        model_weights = '/Users/marissac/caffe/examples/ssd/models/VGGNet/cocoText/SSD_300x300/VGG_cocoText_legibleSplit_SSD_300x300_iter_60000.caffemodel'
        
        model_data_read = '/Users/marissac/caffe/examples/ssd/models/VGGNet/cocoText/SSD_300x300/test_data_read_anno.prototxt'
        
        net = caffe.Net(model_def,model_weights,caffe.TEST)
        net_data = caffe.Net(model_data_read,caffe.TEST)
        #solver.net.copy_from(model_weights)
        #net = caffe.Net(model_def,model_weights,caffe.TRAIN)
        
        ##solver.step(1)
        ##solver.step(1)
        net.forward()
        net_data.forward()
        #
        gt_data = net.blobs['label'].data
        num_gt = gt_data.shape[2]
        prior_data = net.blobs['mbox_priorbox'].data
        num_priors = prior_data.shape[2]/4
        loc_data = net.blobs['mbox_loc'].data
        conf_data = net.blobs['mbox_conf'].data
        num_classes = conf_data.shape[1]/num_priors
        num_batch = loc_data.shape[0]
        max_matches = 20
        overlap_threshold = 0.4        
        
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
        self.top[0] = [caffe.Blob((2,4,20,1))]
        self.top[1] = [caffe.Blob((40,83))]
        self.top[2] = [caffe.Blob((2,80,1,1))]
        self.top[3] = [caffe.Blob((40,33))]
        self.top[4] = [caffe.Blob((40,1))]
        self.top[5] = [caffe.Blob((2,80))]
        #print pred.shape
        #print label.shape

    def test_euclidean(self):
        model_def = '/Users/marissac/caffe/examples/ssd/models/VGGNet/cocoText/SSD_300x300/test_annoWord.prototxt'
        model_weights = '/Users/marissac/caffe/examples/ssd/models/VGGNet/cocoText/SSD_300x300/VGG_cocoText_legibleSplit_SSD_300x300_iter_60000.caffemodel'
        
        model_data_read = '/Users/marissac/caffe/examples/ssd/models/VGGNet/cocoText/SSD_300x300/test_data_read_anno.prototxt'
        
        net = caffe.Net(model_def,model_weights,caffe.TEST)
        net_data = caffe.Net(model_data_read,caffe.TEST)
        #solver.net.copy_from(model_weights)
        #net = caffe.Net(model_def,model_weights,caffe.TRAIN)
        
        ##solver.step(1)
        ##solver.step(1)
        net.forward()
        net_data.forward()
        #
        gt_data = net.blobs['label'].data

        prior_data = net.blobs['mbox_priorbox'].data
        num_priors = prior_data.shape[2]/4
        loc_data = net.blobs['mbox_loc'].data
        conf_data = net.blobs['mbox_conf'].data
              
        
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
        top[0] = [caffe.Blob((2,4,20,1))]
        top[1] = [caffe.Blob((40,83))]
        top[2] = [caffe.Blob((2,80,1,1))]
        top[3] = [caffe.Blob((40,33))]
        top[4] = [caffe.Blob((40,1))]
        top[5] = [caffe.Blob((2,80))]        
        
        lp = caffe_pb2.LayerParameter()
        lp.type = "Python"
        lp.python_param.module = "multiboxRoutingLayer"
        lp.python_param.layer = "MultiboxRoutingLayer"
        lp.python_param.param_str = "{'max_matches': 20,'overlap_threshold': 0.5}"
        layer = caffe.create_layer(lp)
        #layer.SetUp(self.bottom, self.top)
        layer.Reshape(bottom, top)
        layer.Forward(bottom, top)
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
#TestGradientChecker.setUp(testGrad)
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
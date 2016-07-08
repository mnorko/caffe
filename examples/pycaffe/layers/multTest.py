# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 15:56:17 2016

@author: marissac
"""

import caffe 
import numpy as np
import yaml

class multTestLayer(caffe.Layer):
    
    def setup(self, bottom, top):
        """
        Nothing happens
        """
        
        
    def reshape(self,bottom,top):

        N = bottom[0].shape[0] # number of batches
        C = bottom[0].shape[1] # number of channels

       # shape = np.array((N, C, params["output_H"], params["output_W"]))
        top[0].reshape(N,C)
        
        
    def forward(self,bottom,top):
        
        top[0].data[...] = np.multiply(bottom[0].data[...],bottom[0].data[...])
        
    def backward(self, top, propagate_down, bottom):
        
        bottom[0].diff[...] = 2*bottom[0].data*top[0].diff
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 16:21:20 2016

@author: marissac
"""

import caffe 
import numpy as np
import yaml

class BboxToTransformLayer(caffe.Layer):
    
    def setup(self, bottom, top):
        """
        """
        
    def reshape(self,bottom,top):
        box_match_shape = bottom[1].shape[0]
        num_batch_used = bottom[0].shape[0]
        H = bottom[2].shape[2]
        W = bottom[2].shape[3]
        
        top[0].reshape(box_match_shape,6)
        top[1].reshape(num_batch_used,1,H,W)
        top[2].reshape(box_match_shape,23)
        
    def forward(self,bottom,top):
        """
        Convert the bounding boxes to theta values to input into the spatial transform layer
        Convert the images to grayscale and subselect them based on the images that have legible text
        
        Parameters
        ---------
        top: [0] - theta [num_batch_used*max_matches x 6]
             [1] - grayscale image data [num_batch_used x 1 x H x W]
             [2] - gt labels for text reader [num_batch_used*max_matches x 23]
        bottom: [0] - pred_bbox - bounding boxes for each image [num_batch_used x 4 x max_matches x 1]
                [1] - gt_word_labels - contains ground truth information about bbox and the words - sent to word reader part of the net [num_batch_used*max_matches x 33 x 1 x1]
                [2] - data input - images that are padded on the right and bottom to enlarge to some standard size [num_batch,num_channels,H,W]
        """
        pred_bbox = bottom[0].data
        gt_data = bottom[1].data
        img = bottom[2].data
        H = img.shape[2]
        W = img.shape[3]
        num_batch_used = pred_bbox.shape[0]
        max_matches = pred_bbox.shape[2]
        
        # Find the indices for the images in the batch with legible text
        batch_idx_num = gt_data[:,0,0,0]
        batch_idx_used = np.unique(batch_idx_num)
        
        img_output = np.zeros((num_batch_used,1,H,W))         
        theta = np.zeros((num_batch_used*max_matches,6))
        for i in range(0,num_batch_used):
            for k in range(0,max_matches):
                width_orig = gt_data[i*max_matches + k,9,0,0]
                height_orig = gt_data[i*max_matches + k,8,0,0]
                # If we aren't using the pred box then set the theta to the identity matrix
                if width_orig == -1:
                    theta[i*max_matches + k,:] = np.array([1,0,0,0,1,0])
                else:
                    # Pred_bboxes are aon a 0 to 1 scale - convert them to be relative to the new image size
                    xmin = pred_bbox[i,0,k]*width_orig 
                    ymin = pred_bbox[i,1,k]*height_orig
                    xmax = pred_bbox[i,2,k]*width_orig 
                    ymax = pred_bbox[i,3,k]*height_orig
                    
                    bbox_size_x = xmax-xmin
                    bbox_size_y = ymax-ymin
                
                    scale_x = bbox_size_x/W
                    scale_y = bbox_size_y/H
                    
                    bbox_x = 2*(-(W-bbox_size_x)/2 + xmin)/W
                    bbox_y = 2*(-(H-bbox_size_y)/2 + ymin)/H
                    
                    theta[i*max_matches + k,:] = np.array([scale_x,0,bbox_x,0,scale_y,bbox_y])
            img_output[i,0,:,:] = 0.299*img[int(batch_idx_used[i]),0,:,:] +  0.587*img[int(batch_idx_used[i]),1,:,:] + 0.114*img[int(batch_idx_used[i]),2,:,:]

        top[0].data[...] = theta
        top[1].data[...] = img_output
        top[2].data[...] = gt_data[:,10:,0,0]
        
    def backward(self,top,propagate_down,bottom):
        """
        Backpropagate the gradient for theta
        
        Parameters:
        -----------
        top: [0] - theta [num_batch_used*max_matches x 6]
             [1] - grayscale image data [num_batch_used x channel x H x W]
        """
        theta_diff = top[0].diff
        img_data = top[1].data
        num_batch_used = img_data.shape[0]
        max_matches = theta_diff.shape[0]/num_batch_used
        H = bottom[2].shape[2]
        W = bottom[2].shape[3]
        gt_data = bottom[1].data
        
        dPred = np.zeros((num_batch_used,4,max_matches))
        for i in range(0,num_batch_used):
            for k in range(0,max_matches):
                width_orig = gt_data[i*max_matches + k,9,0,0]
                height_orig = gt_data[i*max_matches + k,8,0,0]
                
                diff_temp = theta_diff[i*max_matches + k,:]
                dPred[i,0,k] = -diff_temp[0]*(width_orig/W) + diff_temp[2]*(2/W*(1-width_orig/2))
                dPred[i,1,k] = -diff_temp[4]*(height_orig/H) + diff_temp[5]*(2/H*(1-height_orig/2))
                dPred[i,2,k] = width_orig/W*(diff_temp[0] + diff_temp[2])
                dPred[i,3,k] = height_orig/H*(diff_temp[1]+diff_temp[3])
                
        dPred =  np.expand_dims(dPred,axis = 3)
        bottom[0].diff[...] = dPred
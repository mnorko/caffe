# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 12:09:55 2016

@author: marissac
"""

import caffe 
import numpy as np
import yaml

class SpatialTransformerFastLayer(caffe.Layer):
     """
    Transform an input using the transformation parameters computed by the 
    localization network. This requires the creation of a mesh grid, a 
    transformation of this gird, and an interpolation of the image along the grid
    """
    
    def setup(self, bottom, top):
        params = yaml.load(self.param_str)
        #print params
        #check_params(params)
        
        
    def reshape(self,bottom,top):
        params = yaml.load(self.param_str)
        
        N = bottom[0].shape[0] # number of batches
        C = bottom[0].shape[1] # number of channels
        
       # shape = np.array((N, C, params["output_H"], params["output_W"]))
        top[0].reshape(N,C,params["output_H"], params["output_W"])
        
        
    def forward(self,bottom,top):
        """
        Transform the mesh grid
        
        Paremters
        ---------
        top: [0] - bounding boxes for each image [num_batch x 4 x num_possible_matches x 1]
             [1] - ground truth labels [(num_possible_matches*num_batches) x 23 x 1 x 1]
        bottom: [0] - mbox_loc input [num_batches x (num_priorbox*4)] the four coordinates represnt xmin,ymin,xmax,ymax
                [1] - mbox_conf input [num_batches x (num_priorbox*num_classes)] 
                [2] - mbox_priorbox input [1 x 2 x (num_priorbox*4)] where the dimension with 2 represents location in one column and variance in the other
                [3] - label [1 x 1 x num_gt_detects x 8] - the 8 coords represent [item_id, group_label, instance_id, xmin, ymin, xmax, ymax, diff] 
        """
        gt_data = bottom[3].data
        num_gt = gt_data.shape[2]
        prior_data = bottom[2].data
        num_priors = prior_data.shape[2]/4
        loc_data = bottom[0].data
        conf_data = bottom[1].data
        num_batch = loc_data.shape[0]
        num_classes = conf_data.shape[1]/num_priors
        
        # Retrieve bounding boxes from the ground truth boxes
        all_gt_bboxes = multibox_util.getGroundTruth(gt_data,num_gt)
        
        # Retreive prior box bounding box and variance information
        prior_bboxes, prior_variances = multibox_util.getPriorBBoxes(prior_data,num_priors)
        
        # Retrieve corrections that should be made to the prior boxes for each prediction
        all_loc_preds = multibox_util.getLocPredictions(loc_data,num_batch,num_priors,num_classes)
        
        # Find matches between the predictions and ground turht boxes
        for i in range(0,num_batch):
            loc_bboxes = multibox_util.decodeBBoxes(prior_bboxes,prior_variances,all_loc_preds[i])
            match_indices, match_overlaps = multibox_util.matchBBox(gt_bboxes,loc_bboxes,overlap_threshold)
            temp_matches_indices,temp_match_overlap = multibox_util.MatchBBox(all_gt_bboxes,prior_bboxes,overlap_threshold)
        
    def backward(self, top, propagate_down, bottom):
        """
        Backpropagate the gradients for theta and the input U
        Parameters
        ---------
        top: information from the above layer with the transformed image
        bottom: [0] - input image or convolutional layer input [num_batch x num_channels x input_height x input_width]
                [1] - theta parameters [num_batch x Num_transform_param] - in this case the num_transform_param = 6
        """
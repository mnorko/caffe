# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 12:09:55 2016

@author: marissac
"""

import caffe 
import numpy as np
import yaml
import multibox_util

class MultiboxRoutingLayer(caffe.Layer):
    
    def setup(self, bottom, top):
        params = yaml.load(self.param_str)
        
        
    def reshape(self,bottom,top):
        params = yaml.load(self.param_str)
        max_matches = params["max_matches"]
        
        gt_data = bottom[3].data

        # Retrieve bounding boxes from the ground truth boxes
        batch_idx_num = gt_data[0,0,:,0]
        global batch_with_words
        batch_with_words = np.unique(batch_idx_num)
        num_batch_used = len(batch_with_words)
        
        num_priors = bottom[2].data.shape[2]/4
        num_classes = bottom[1].data.shape[1]/num_priors
        num_labels = bottom[3].shape[3]      
        
       # shape = np.array((N, C, params["output_H"], params["output_W"]))
        top[0].reshape(num_batch_used,4,max_matches,1)
        top[1].reshape(num_batch_used*max_matches,num_classes)
        top[2].reshape(num_batch_used,4*max_matches)
        top[3].reshape(num_batch_used*max_matches,num_labels,1,1)
        top[4].reshape(num_batch_used*max_matches)
        top[5].reshape(num_batch_used,4*max_matches)
        
        
    def forward(self,bottom,top):
        """
        Find matching ground truth boxes and predicted boxes
        Select the correct bounding boxes and ground truth labels to send to the next layer
        
        Paremters
        ---------
        top: [0] - pred_box_final - bounding boxes for each image [num_batch_used x 4 x max_matches x 1]
             [1] - conf_final - confidence scores in all of the classes [num_batch_used*max_matches x num_classes]
             [2] - loc_data_final - contains bounding box information in a different format than pred_box_final - this goes to the localization outut [num_batches_used*num_matches_used x 4]
             [3] - labels_final - contains ground truth information about bbox and the words - sent to word reader part of the net [num_batch_used*max_matchs x 33]
             [4] - class_labels - contains the true label for each ground truth detection - in this case, all of the values should be the label for legible text [num_batches_used*max_matches]
             [5] - encode_gt_bboxes_final - this contains the ground truth bounding box data encoded so it can be compared to the mbox_loc_data - sent to localization loss layer [num_batches_used*num_matches_used x 4]
        bottom: [0] - mbox_loc input [num_batches x (num_priorbox*4)] the four coordinates represnt xmin,ymin,xmax,ymax
                [1] - mbox_conf input [num_batches x (num_priorbox*num_classes)] 
                [2] - mbox_priorbox input [1 x 2 x (num_priorbox*4)] where the dimension with 2 represents location in one column and variance in the other
                [3] - label [1 x 1 x num_gt_detects x 33] - the 31 coords represent [item_id, group_label, instance_id, xmin, ymin, xmax, ymax, diff, img_height, img_width, 23 characers for word] 
        """
        params = yaml.load(self.param_str)
        max_matches = params["max_matches"]
        overlap_threshold = params["overlap_threshold"]
        
        gt_data = bottom[3].data
        num_gt = gt_data.shape[2]
        prior_data = bottom[2].data
        num_priors = prior_data.shape[2]/4
        loc_data = bottom[0].data
        conf_data = bottom[1].data
        num_batch = loc_data.shape[0]
        num_classes = conf_data.shape[1]/num_priors
        # Need to read overlap_threshold and max_matches from parameters
        
        
        # Retrieve bounding boxes from the ground truth boxes
        all_gt_bboxes, gt_batch_id = multibox_util.getGroundTruth(gt_data,num_gt,num_batch)
        
        # Find the number of batches with label information
        
        num_batch_used = len([x for x in gt_batch_id if len(x) >0])
        
        global prior_bboxes
        global prior_variances
        # Retreive prior box bounding box and variance information
        prior_bboxes, prior_variances = multibox_util.getPriorBBoxes(prior_data,num_priors)
        
        global all_loc_preds
        # Retrieve corrections that should be made to the prior boxes for each prediction
        all_loc_preds = multibox_util.getLocPredictions(loc_data,num_batch,num_priors)
        
        # Get the confidence for each class - in this case, we're setting up the layer assuming there's only one class
        all_max_scores, all_conf_scores = multibox_util.getMaxConfidenceScores(conf_data,num_batch,num_priors,num_classes,0)
        all_conf_scores = np.reshape(conf_data,(num_batch,num_priors,num_classes))
        
        global match_indices_final
        match_indices_final = np.zeros((num_batch_used ,max_matches))
        match_gt_indices_final = np.zeros((num_batch_used ,max_matches))
        pred_box_final = np.zeros((num_batch_used ,4,max_matches))
        #encode_gt_bboxes_final = np.zeros((num_batch_used ,4*max_matches))
        #loc_data_final = np.zeros((num_batch_used ,4*max_matches))
        batch_idx_use = []
        batch_count = 0
        # Find matches between the predictions and ground truth boxes
        for i in range(0,num_batch):
            if len(all_gt_bboxes[i]) > 0:
                # Decode bboxes by incorporating the location deltas
                loc_bboxes = multibox_util.decodeBBoxes(prior_bboxes,prior_variances,all_loc_preds[i])
                # Find the predicted bounding boxes that match the ground truth and high confidence predictions
                match_indices, match_gt_indices, match_overlaps =multibox_util.createBBoxMatchSet(all_gt_bboxes[i],loc_bboxes,all_max_scores[i,:],overlap_threshold,max_matches)
                # Subselect the predicted bboxes which represent where the objects are in the real image        
                pred_final_bboxes = multibox_util.subselectBBoxes(loc_bboxes,match_indices)
                # Subselect the labels that give all the ground truth information
                batch_labels = multibox_util.selectLabels(gt_data,match_gt_indices,gt_batch_id[i])
                # Subselect the confidences to send to the confidence loss layer
                conf_match_scores = multibox_util.subselectConf(all_conf_scores[i,:,:],match_indices)
                # Encode the ground truth bboxes to they can be compared to the mbox_loc data
                encode_gt_bboxes = multibox_util.encodeBBoxes(prior_bboxes,prior_variances,all_gt_bboxes[i],match_gt_indices,match_indices)
                # Reshape and select mbox_loc data that can be output to the localization loss layer
                loc_data_all = np.reshape(loc_data[i,:],(num_priors,4))
                good_indices = np.where(match_indices!=-1)
                match_indices_cut = np.int32(match_indices[good_indices])
                loc_data_subselect = loc_data_all[match_indices_cut,:] # After this line, loc_data_subselect is max_matchesx4
                #loc_data_subselect = np.reshape(loc_data_subselect,[-1])
                
                # Accumulate the data for each batch
                pred_box_final[batch_count,:,:] = pred_final_bboxes
                match_indices_final[batch_count,:] = match_indices
                match_gt_indices_final[batch_count,:] = match_gt_indices
                #encode_gt_bboxes_final[batch_count,:] = encode_gt_bboxes[:,0]
                #loc_data_final[batch_count,:] = loc_data_subselect
                if batch_count == 0:
                    labels_final = batch_labels
                    conf_final = conf_match_scores
                    loc_data_final = loc_data_subselect
                    encode_gt_bboxes_final = encode_gt_bboxes
                else:
                    labels_final = np.concatenate((labels_final,batch_labels),axis = 0)
                    conf_final = np.concatenate((conf_final,conf_match_scores),axis = 0)
                    loc_data_final = np.concatenate((loc_data_final,loc_data_subselect),axis = 0)
                    encode_gt_bboxes_final = np.concatenate((encode_gt_bboxes_final,encode_gt_bboxes),axis=0)
                    
                batch_idx_use.append(i)
                batch_count = batch_count +1
        

        pred_box_final = np.expand_dims(pred_box_final,axis = 3)
        class_labels = labels_final[:,1]
        labels_final = np.expand_dims(labels_final,axis=2)
        labels_final = np.expand_dims(labels_final,axis=3)
        #loc_data_final = np.expand_dims(loc_data_final,axis = 2)
        #loc_data_final = np.expand_dims(loc_data_final,axis=3)
        
        top[2].reshape(loc_data_final.shape[0],loc_data_final.shape[1])
        top[5].reshape(encode_gt_bboxes_final.shape[0],encode_gt_bboxes_final.shape[1])
        top[0].data[...] = pred_box_final # Goes to spatial transformer net
        top[1].data[...] = conf_final # Goes to confidence loss layer
        top[2].data[...] = loc_data_final # GOes to localization loss layer
        top[3].data[...] = labels_final # Goes to word reader loss layer
        top[4].data[...] = class_labels # Goes to confidence loss layer
        top[5].data[...] = encode_gt_bboxes_final # Goes to localization loss layer
        
        
    def backward(self, top, propagate_down, bottom):
        """
        Backpropagate the gradients for mbox_loc and mbox_conf
        Parameters
        ---------
        top[0] contains information about the predicted location [num_batches_used x 4 x max_matches x 1] where the 4 coordinates are xmin, ymin, xmax, ymax
        top[1] contains information about the confidence [(num_batchs_used*max_matches)xnum_classes]
        top[2] contains gradients on predictions locations from the localization loss [num_batches_used x (4*max_matches)]
        """
        
        # Make sure the gradients are already weighted by the loss weight
        # Get prior_bboxes and prior_variances in this function
        top_diff = top[0].diff
        conf_diff = top[1].diff
        loc_diff = top[2].diff
        loc_data = bottom[0].data
        num_priors = loc_data.shape[1]/4
        num_batch = loc_data.shape[0]
        max_matches = top_diff.shape[2]
        # Get num_priors defined in here
        dLoc = np.zeros((num_batch,4,num_priors))
        # Get the gradient for mbox_loc
        loc_count = 0
        for j in range(0,len(batch_with_words)):
            batch_idx = int(batch_with_words[j])
            for i in range(0,max_matches):
                match_idx_use = int(match_indices_final[j,i])
                if match_idx_use != -1:
                
                    #print match_idx_use
                    w_prior = prior_bboxes[match_idx_use]['xmax'] - prior_bboxes[match_idx_use]['xmin'] 
                    h_prior = prior_bboxes[match_idx_use]['ymax'] - prior_bboxes[match_idx_use]['ymin']
                    localization_grad = loc_diff[loc_count,:]
                    loc_count = loc_count +1
                    # Gradient for xmin
    
                    dLoc[batch_idx,0,match_idx_use] = dLoc[batch_idx,0,match_idx_use] + (top_diff[j,0,i,0]+top_diff[j,2,i,0])*prior_variances[match_idx_use]['xmin']*w_prior + localization_grad[0]
                    # Gradient for ymin
                    dLoc[batch_idx,1,match_idx_use] = dLoc[batch_idx,1,match_idx_use] + (top_diff[j,1,i,0]+top_diff[j,3,i,0])*prior_variances[match_idx_use]['ymin']*h_prior + localization_grad[1]
                    # Gradient for xmax
                    dLoc[batch_idx,2,match_idx_use] = dLoc[batch_idx,2,match_idx_use] + prior_variances[match_idx_use]['xmax']*np.exp(prior_variances[match_idx_use]['xmax']*all_loc_preds[batch_idx][match_idx_use]['xmax'])*w_prior/2*(top_diff[j,2,i,0]-top_diff[j,0,i,0]) + localization_grad[2]
                    # Gradient for ymax
                    dLoc[batch_idx,3,match_idx_use] = dLoc[batch_idx,3,match_idx_use] + prior_variances[match_idx_use]['ymax']*np.exp(prior_variances[match_idx_use]['ymax']*all_loc_preds[batch_idx][match_idx_use]['ymax'])*h_prior/2*(top_diff[j,3,i,0]-top_diff[j,1,i,0]) + localization_grad[3]
        #Reshape dLoc so it's num_batchx(4*num_priors)
        dLoc = np.reshape(dLoc,(num_batch,4*num_priors))        
        # THe backpropagation for the confidence is just routed from the top layer
        dConf = multibox_util.sortConfGrad(conf_diff,match_indices_final,num_priors,num_batch,batch_with_words)
            
        if propagate_down[0]: # Propagate convolutional input gradients down
            bottom[0].diff[...] = dLoc
            
        if propagate_down[1]: # Propagate theta gradients down
            bottom[1].diff[...] = dConf
        
        
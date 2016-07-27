# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 13:15:56 2016

@author: marissac
"""

import numpy as np
import math
import caffe



def getGroundTruth(gt_data,num_gt,num_batch):
    """
    Retrieve bounding boxes from ground truth labels

    Parameters
    -----------
    gt_data: ground truth labels [ 1 x 1 x num_gt_detects x 8] - the 8 coords represent [item_id, group_label, instance_id, xmin, ymin, xmax, ymax, difficult_flag]
    item_id indicates which image in the batch it corresponds to
    """
    bbox = []
    batch_id = []
    for k in range(0,num_batch):
        bbox_temp = []
        batch_id_temp = []
        for i in range(0,num_gt):
            if gt_data[0,0,i,0] == k:
                bbox_temp.append({'label': gt_data[0,0,i,1],'xmin': gt_data[0,0,i,3],'ymin': gt_data[0,0,i,4],
                    'xmax': gt_data[0,0,i,5],'ymax': gt_data[0,0,i,6]})
                batch_id_temp.append(i)
        bbox.append(bbox_temp)
        batch_id.append(batch_id_temp)
    return bbox,batch_id
        

def getPriorBBoxes(prior_data,num_priors):
    """
    Retrieve prior boxes

    Parameters
    ----------
    prior_data: [1 x 2 x (num_priorbox*4)] where the dimension with 2 represents location in one column and variance in the other
    """
    bbox = []
    prior_var = []
    for i in range(0,num_priors):
        start_idx = i * 4
        bbox.append({'xmin':prior_data[0,0,start_idx],'ymin':prior_data[0,0,start_idx+1],
            'xmax':prior_data[0,0,start_idx+2],'ymax':prior_data[0,0,start_idx+3]})
        prior_var.append({'xmin':prior_data[0,1,start_idx],'ymin':prior_data[0,1,start_idx+1],
            'xmax':prior_data[0,1,start_idx+2],'ymax':prior_data[0,1,start_idx+3]})
    return bbox, prior_var


def getLocPredictions(loc_data,num_batch,num_priors):
    """ 
    Retrieve predicted bounding box corrections to the prior boxes

    Parameters
    ----------
    loc_data: [num_batches x (num_priorbox*4)] the four coordinates represnt xmin,ymin,xmax,ymax

    """
    bbox_total = []
    for k in range(0,num_batch):
        bbox = []
        for i in range(0,num_priors):
            start_idx = i * 4
            bbox.append({'xmin':loc_data[k,start_idx],'ymin':loc_data[k,start_idx+1],
                'xmax':loc_data[k,start_idx+2],'ymax':loc_data[k,start_idx+3]})
        bbox_total.append(bbox)
    return bbox_total

def getMaxConfidenceScores(conf_data,num_batch, num_priors,num_classes,background_label_id):
    """ 
    Retrieve the maximum class confidences for each sample. 
    For the router label, we will assume there is only a single class

    Parameters
    ----------
    conf_data: [num_batches x (num_priorbox*num_classes)] 

    """
    all_max_scores = np.zeros((num_batch,num_priors))
    all_conf_scores = np.zeros((num_batch,num_priors,num_classes))
    for k in range(0,num_batch):
        for i in range(0,num_priors):
            start_idx = i*num_classes
            maxconf = np.array((-1000000))
            softmax_sum = 0
            for c in range(0,num_classes):
                # Only find confidence if the class is not the background class
                if c != background_label_id:
                    maxconf = max(conf_data[k,start_idx+c],maxconf)
                softmax_sum = softmax_sum + math.exp(conf_data[k,start_idx + c])
            all_max_scores[k,i] = math.exp(maxconf)/softmax_sum
    return all_max_scores, all_conf_scores


def intersectBBox(bbox1,bbox2):
    """
    Find the amount of intersection between two bounding boxes

    Parameters
    ----------
    bbox1: dictionary with with the following keys: 'label','xmin','xmax','ymin','ymax'
    bbox2: dictionary with with the following keys: 'label','xmin','xmax','ymin','ymax'

    """
    intersect_bbox = {}
    if((bbox2['xmin'] > bbox1['xmax']) | (bbox2['xmax'] < bbox1['xmin']) |
        (bbox2['ymin'] > bbox1['ymax']) | (bbox2['ymax'] < bbox1['ymin'])):
        intersect_bbox['xmin'] = 0
        intersect_bbox['ymin'] = 0
        intersect_bbox['xmax'] = 0
        intersect_bbox['ymax'] = 0
    else:
        intersect_bbox['xmin'] = max(bbox1['xmin'],bbox2['xmin'])
        intersect_bbox['ymin'] = max(bbox1['ymin'],bbox2['ymin'])
        intersect_bbox['xmax'] = min(bbox1['xmax'],bbox2['xmax'])
        intersect_bbox['ymax'] = min(bbox1['ymax'],bbox2['ymax'])
    return intersect_bbox

def BBoxSize(bbox):
    width = bbox['xmax'] - bbox['xmin']
    height = bbox['ymax']-bbox['ymin']
    return width*height

def JaccardOverlap(bbox1,bbox2):
    intersect_bbox = intersectBBox(bbox1,bbox2)
    intersect_width = intersect_bbox['xmax']-intersect_bbox['xmin'] 
    intersect_height = intersect_bbox['ymax'] - intersect_bbox['ymin'] 
    intersect_size = intersect_width * intersect_height
    bbox1_size = BBoxSize(bbox1)
    bbox2_size = BBoxSize(bbox2)
    return intersect_size/ (bbox1_size + bbox2_size - intersect_size)


def decodeBBox(prior_box,prior_variances,bbox):
    """
    Adjust a single prior box using the predicted offsets

    Parameters
    ----------
    prior_box, prior_variance,s bbox: dictionaries with the following keys: 'label','xmin','xmax','ymin','ymax'

    """
    prior_width = prior_box['xmax'] - prior_box['xmin']
    prior_height = prior_box['ymax'] - prior_box['ymin']
    prior_center_x = (prior_box['xmax'] + prior_box['xmin'])/2.0
    prior_center_y = (prior_box['ymax'] + prior_box['ymin'])/2.0

    decode_bbox_center_x = prior_variances['xmin']*bbox['xmin']*prior_width+prior_center_x
    decode_bbox_center_y = prior_variances['ymin']*bbox['ymin']*prior_height+prior_center_y
    decode_bbox_width = np.exp(prior_variances['xmax']*bbox['xmax'])*prior_width
    decode_bbox_height = np.exp(prior_variances['ymax']*bbox['ymax'])*prior_height

    decode_bbox = {}
    decode_bbox['xmin'] = np.clip(decode_bbox_center_x - decode_bbox_width/2.0,0,1)
    decode_bbox['ymin'] = np.clip(decode_bbox_center_y - decode_bbox_height/2.0,0,1)
    decode_bbox['xmax'] = np.clip(decode_bbox_center_x + decode_bbox_width/2.0,0,1)
    decode_bbox['ymax'] = np.clip(decode_bbox_center_y + decode_bbox_height/2.0,0,1)
    return decode_bbox

def decodeBBoxes(prior_bboxes,prior_variances,bboxes):
    """
    Convert the bounding boxes to take into account the predicted offsets

    Parameters
    ----------
    prior_bboxes: list (length num_priors) of dictionaries. Each dictionary contains prior box label and locations ('label','xmin','xmax','ymin','ymax')
    prior_variances: list (length num_priors) of dictionaries. Each dictionary contains prior box label and variance ('label','xmin','xmax','ymin','ymax')
    bboxes: list (length num_priors) of dictionaries containing the predicted offsets for the prior box locations ('label','xmin','xmax','ymin','ymax')

    """
    decode_bboxes = []
    num_bboxes = len(prior_bboxes)
    for i in range(0,num_bboxes):
        decode_bbox_temp = decodeBBox(prior_bboxes[i],prior_variances[i],bboxes[i])
        decode_bboxes.append(decode_bbox_temp)
    return decode_bboxes

def encodeBBoxes(prior_bboxes,prior_variances,gt_bboxes,match_gt_indices,match_indices):
    """
    Encode the matched ground truth bboxes so they can be sent to the smooth L1 Loss Layer

    Parameters:
    -----------
    prior_bboxes: list (length num_priors) of dictionaries. Each dictionary contains prior box label and locations ('label','xmin','xmax','ymin','ymax')
    prior_variances: list (length num_priors) of dictionaries. Each dictionary contains prior box label and variance ('label','xmin','xmax','ymin','ymax')
    gt_bboxes: list (length ground truth labels) of dictionaries. Each dictionary contains ground truth and locations ('label','xmin','xmax','ymin','ymax')
    match_gt_indices: numpy array [max_matches]

    Outputs:
    --------
    encode_bbox: contains encoded ground truth bounding boxes [num_matches_used x 4] where each set of four represents xmin, ymin, xmax, ymax
    """
    max_matches = match_gt_indices.shape[0]
    for k in range(0,max_matches):
        match_idx_use = int(match_indices[k])
        match_gt_idx_use = int(match_gt_indices[k])
        if match_idx_use != -1:
              prior_width = prior_bboxes[match_idx_use]['xmax']-prior_bboxes[match_idx_use]['xmin']
              prior_height = prior_bboxes[match_idx_use]['ymax']-prior_bboxes[match_idx_use]['ymin']
              prior_center_x = (prior_bboxes[match_idx_use]['xmax']+prior_bboxes[match_idx_use]['xmin'])/2.0
              prior_center_y = (prior_bboxes[match_idx_use]['ymax']+prior_bboxes[match_idx_use]['ymin'])/2.0
            
              bbox_width = gt_bboxes[match_gt_idx_use]['xmax'] - gt_bboxes[match_gt_idx_use]['xmin'] 
              bbox_height = gt_bboxes[match_gt_idx_use]['ymax'] - gt_bboxes[match_gt_idx_use]['ymin'] 
              bbox_center_x = (gt_bboxes[match_gt_idx_use]['xmax'] + gt_bboxes[match_gt_idx_use]['xmin'])/2.0
              bbox_center_y = (gt_bboxes[match_gt_idx_use]['ymax'] + gt_bboxes[match_gt_idx_use]['ymin'])/2.0
            
              xmin = (bbox_center_x-prior_center_x)/prior_width/prior_variances[match_idx_use]['xmin']
              ymin = (bbox_center_y-prior_center_y)/prior_height/prior_variances[match_idx_use]['ymin']
              xmax = np.log(bbox_width/prior_width)/prior_variances[match_idx_use]['xmax']
              ymax = np.log(bbox_height/prior_height)/prior_variances[match_idx_use]['ymax']
            
              encode_add = np.array([[xmin],[ymin],[xmax],[ymax]])
              if k == 0:
                  encode_bbox = encode_add
              else:
                  encode_bbox = np.concatenate((encode_bbox,encode_add),axis = 0)
    num_match_used = encode_bbox.shape[0]/4
    encode_bbox = encode_bbox.reshape(num_match_used,4)
    return encode_bbox

def createBBoxMatchSet(gt_bboxes,pred_bboxes,confidence,overlap_threshold,max_matches):
    """
    Match the ground truth boxes and the corrected prior boxes using IOU
    
    Parameters
    ----------
    gt_bboxes: list (length ground truth labels) of dictionaries. Each dictionary contains ground truth and locations ('label','xmin','xmax','ymin','ymax')
    prior_bboxes: list (length num_priors) of dictionaries. Each dictionary contains prior box label and locations ('label','xmin','xmax','ymin','ymax')
    confidence: numpy array [num_priors]
    overlap_threshold: iou threshold required to declare a match

    """
    num_pred = len(pred_bboxes)
    num_gt = len(gt_bboxes)
    overlap = np.zeros((num_pred,num_gt))
    for i in range(0,num_pred):
        for j in range(0,num_gt):
            overlap[i,j] = JaccardOverlap(pred_bboxes[i],gt_bboxes[j])

    # Find the best match for each ground truth 
    match_gt_indices = []
    match_indices = []
    match_overlap = []
    match_conf = []
    new_matches_overlap = []
    new_matches_gt_idx = []
    new_matches_idx = []
    new_matches_conf = []
    for j in range(0,num_gt):
        max_overlap = 0
        max_idx = -1
        max_gt_idx = -1
        for i in range(0,num_pred):
            if overlap[i,j] > max_overlap:
                max_idx = i
                max_gt_idx = j
                max_overlap = overlap[i,j]

        if max_idx == -1:
            break
        else:
            match_gt_indices.append(max_gt_idx)
            match_indices.append(max_idx)
            match_overlap.append(max_overlap)
            match_conf.append(confidence[max_idx])
            # Zero out the overlap so it won't be picked up in the next loop
            overlap[i,j] = 0
        #Get the remaining overlaps > 0.5 
        for i in range(0,num_pred):
            if overlap[i,j] > overlap_threshold:
                new_matches_overlap.append(overlap[i,j])
                new_matches_gt_idx.append(j)
                new_matches_idx.append(i)
                new_matches_conf.append(confidence[i])

    # Convert lists to numpy arrays
    match_indices = np.array(match_indices)
    match_gt_indices = np.array(match_gt_indices)
    match_overlap = np.array(match_overlap)
    match_conf = np.array(match_conf)
    new_matches_overlap = np.array(new_matches_overlap)
    new_matches_gt_idx = np.array(new_matches_gt_idx)
    new_matches_idx = np.array(new_matches_idx)
    new_matches_conf = np.array(new_matches_conf)

    # Pick the matches with > 0.5 overlap with the highest confidence predictions
    # First sort all the remaining overlapping matches based on confidence
    # Contains list of idx for entries in ascending order
    sort_idx = np.argsort(new_matches_conf)


    num_add_matches = max_matches - match_indices.shape[0]

    # Check that there are enough entries to add
    num_new_matches = sort_idx.shape[0]

    # Get the indices for the top matches
    if num_new_matches >= num_add_matches:
        top_matches_idx = sort_idx[(num_new_matches-num_add_matches):]
        match_indices= np.concatenate((match_indices,new_matches_idx[top_matches_idx]),axis = 0)
        match_gt_indices= np.concatenate((match_gt_indices,new_matches_gt_idx[top_matches_idx]),axis = 0)
        match_overlap = np.concatenate((match_overlap,new_matches_overlap[top_matches_idx]),axis = 0)
        match_conf = np.concatenate((match_conf,new_matches_conf[top_matches_idx]),axis = 0)
    else:
        match_indices = np.concatenate((match_indices,new_matches_idx,-1*np.ones((num_add_matches-num_new_matches))),axis = 0)
        match_gt_indices = np.concatenate((match_gt_indices,new_matches_gt_idx,-1*np.ones((num_add_matches-num_new_matches))),axis = 0)
        match_overlap = np.concatenate((match_overlap,new_matches_overlap,np.zeros((num_add_matches-num_new_matches))),axis = 0)
        match_conf = np.concatenate((match_conf,new_matches_conf,np.zeros((num_add_matches-num_new_matches))),axis = 0)
    match_indices = np.int32(match_indices)
    match_gt_indices = np.int32(match_gt_indices)
    return match_indices, match_gt_indices, match_overlap

def subselectBBoxes(bboxes,bbox_idx):
    """
    Select the items in bboxes that correspond to the bbox_idx. 
    There may be multiple instances of each bbox in bbox_idx

    Parameters:
    -----------
    bboxes: list of dictionaries where the indices in the lsit correspond to the indices in bbox_idx
    bbox_idx: numpy array that is the length of max matches
    
    Outputs:
    --------
    bbox_out: numpy array [4 x max_matches] 
    """
    max_matches = bbox_idx.shape[0]
    bbox_out = np.zeros((4,max_matches))
    for k in range(0,max_matches):
        if bbox_idx[k] == -1:
            bbox_out[0,k] = 0
            bbox_out[1,k] = 0
            bbox_out[2,k] = 0.1
            bbox_out[3,k] = 0.1
        else:
            idx_use = int(bbox_idx[k])
            bbox_out[0,k] = bboxes[idx_use]['xmin']
            bbox_out[1,k] = bboxes[idx_use]['ymin']
            bbox_out[2,k] = bboxes[idx_use]['xmax']
            bbox_out[3,k] = bboxes[idx_use]['ymax']
    return bbox_out

def selectLabels(gt_labels,match_gt_indices,batch_id_use):
    """
    Select the ground truth labels to be sent to the next layer.
    In the word reader network, each label word needs a label

    Parameters:
    -----------
    gt_labels: numpy array [1 x 1 x num_gt_detects x 33] - the 33 coords represent [item_id, group_label, instance_id, xmin, ymin, xmax, ymax, diff, img_height, img_width23 characers for word] 
    match_gt_indices: numpy array [max_matches] of indices for the ground truth labels associated with matches going to the next layer
    """
    batch_id_use = np.asarray(batch_id_use)
    gt_labels = gt_labels[:,:,batch_id_use,:]
    max_matches = match_gt_indices.shape[0]
    label_out = np.zeros((max_matches,33))
    for k in range(0,max_matches):
        if match_gt_indices[k] == -1:
            label_out[k,:] = -1*np.ones((1,33)) 
        else:
            label_out[k,:] = gt_labels[0,0,match_gt_indices[k],:]

    return label_out

def subselectConf(conf_scores,match_indices):
    """
    Subselect the confidence scores based on the matches

    Parameters:
    -----------
    conf_scores: numpy array [num_priors x num_classes] - confidence scores for each class for each priorbox
    match_indeices: numpy array [max_matches]
    """
    max_matches = match_indices.shape[0]
    num_classes = conf_scores.shape[1]
    conf_out = np.zeros((max_matches,num_classes))
    for k in range(0,max_matches):
        if match_indices[k] == -1:
            conf_out[k,:] = -1*np.ones((1,num_classes))
        else:
            conf_out[k,:] = conf_scores[match_indices[k],:]

    return conf_out

def sortConfGrad(conf_diff,match_indices,num_priors,num_batch,batch_with_words):
    """
    Reroute the confidence gradients down so they correspond to the matching indices

    Parameters:
    -----------
    conf_diff: gradients for confidence values [(num_batches_used*max_matches)xnum_classes]
    match_indices: numpy array [num_batch_used x max_matches]

    Output:
    -------
    conf_diff_routed: gradient routed to all the correct priorbox indices [num_batchesx(num_priors*num_classes)]
    """
    max_matches = match_indices.shape[1]
    num_classes = conf_diff.shape[1]
    num_batch_used = match_indices.shape[0]
    conf_diff_routed = np.zeros((num_batch,num_priors*num_classes))
    for i in range(0,num_batch_used):
        for k in range(0,max_matches):
            batch_idx = int(batch_with_words[i])
            match_idx = int(match_indices[i,k])
            if match_idx != -1:
                conf_diff_routed[batch_idx,(match_idx*num_classes):num_classes*(match_idx+1)] = conf_diff_routed[batch_idx,(match_idx*num_classes):num_classes*(match_idx +1)] + conf_diff[(i*max_matches + k),:]
            

    return conf_diff_routed





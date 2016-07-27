# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 10:54:38 2016

@author: marissac
"""
# Test the solver for MNIST with the spatial transformer
import numpy as np
import matplotlib.pyplot as plt
import lmdb
from PIL import Image
from StringIO import StringIO
import sys
sys.path.append("/Users/marissac/caffe/examples/pycaffe/layers")

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Make sure that caffe is on the python path:
caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
import os
os.chdir(caffe_root)

import caffe
import multibox_util

from google.protobuf import text_format
from caffe.proto import caffe_pb2

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
overlap_threshold = 0.5

# Retrieve bounding boxes from the ground truth boxes
all_gt_bboxes, gt_batch_id = multibox_util.getGroundTruth(gt_data,num_gt,num_batch)

# Find the number of batches with label information
num_batch_use = len([x for x in gt_batch_id if len(x) >0])

# Retreive prior box bounding box and variance information
prior_bboxes, prior_variances = multibox_util.getPriorBBoxes(prior_data,num_priors)

# Retrieve corrections that should be made to the prior boxes for each prediction
all_loc_preds = multibox_util.getLocPredictions(loc_data,num_batch,num_priors)

# Get the confidence for each class - in this case, we're setting up the layer assuming there's only one class
all_max_scores, all_conf_scores = multibox_util.getMaxConfidenceScores(conf_data,num_batch,num_priors,num_classes,0)
all_conf_scores = np.reshape(conf_data,(num_batch,num_priors,num_classes))

match_indices_final = np.zeros((num_batch_use ,max_matches))
match_gt_indices_final = np.zeros((num_batch_use ,max_matches))
pred_box_final = np.zeros((num_batch_use ,4,max_matches))
#encode_gt_bboxes_final = np.zeros((num_batch_use ,4*max_matches))
#loc_data_final = np.zeros((num_batch_use ,4*max_matches))
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

img = net_data.blobs['data'].data



pred_bbox = pred_box_final
gt_data = labels_final
num_batch = img.shape[0]
H = img.shape[2]
W = img.shape[3]
num_batch_used = pred_bbox.shape[0]
max_matches = pred_bbox.shape[2]

# Find the vindices for the images in the batch with legible text
batch_idx_num = gt_data[:,0,0,0]
batch_idx_used = np.unique(batch_idx_num)

img_output = np.zeros((num_batch_used,1,H,W))         
theta = np.zeros((num_batch_used*max_matches,6))
for i in range(0,num_batch_used):
    for k in range(0,max_matches):
        width_orig = gt_data[i*max_matches + k,9,0,0]
        height_orig = gt_data[i*max_matches + k,8,0,0]
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
    img_output[i,0,:,:] = 0.299*img[batch_idx_used[i],0,:,:] +  0.587*img[batch_idx_used[i],1,:,:] + 0.114*img[batch_idx_used[i],2,:,:]

model_word_def = '/Users/marissac/caffe/examples/ocr/90ksynth/deploy_ocr_spatial_transform_fullTest.prototxt'
model_word_weights = '/Users/marissac/caffe/examples/ocr/90ksynth/90ksynth_v2_v00_iter_140000.caffemodel'


net_word = caffe.Net(model_word_def,      # defines the structure of the model
                model_word_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

net_word.blobs['theta'].data[...] = theta
net_word.blobs['data'].data[...] = img_output

net_word.forward()      


#colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
#img_num = 0
#img_use = net.blobs['data'].data[img_num,1,:,:]
#plt.figure()
#plt.imshow(img_use)
#currentAxis = plt.gca()
#pred_num = 0
#for k in range(0,6):
#    xmin = int(round(pred_box_final[pred_num,0,k,0]*300))
#    ymin = int(round(pred_box_final[pred_num,1,k,0]*300))
#    xmax = int(round(pred_box_final[pred_num,2,k,0]*300))
#    ymax = int(round(pred_box_final[pred_num,3,k,0]*300))
#    coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
#    color = colors[k % len(colors)]
#    name = '%d'%k
#    currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
#    currentAxis.text(xmin, ymin, name, bbox={'facecolor':'white', 'alpha':0.5})

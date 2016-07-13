# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 13:15:56 2016

@author: marissac
"""

import numpy as np
import caffe



def getGroundTruth(gt_data,num_gt):
	"""
	Retrieve bounding boxes from ground truth labels

	Parameters
	-----------
	gt_data: ground truth labels [ 1 x 1 x num_gt_detects x 8] - the 8 coords represent [item_id, group_label, instance_id, xmin, ymin, xmax, ymax, difficult_flag]
	"""
	bbox = []
	for i in range(0,num_gt):
		bbox.append({'label': gt_data[0,0,i,1],'xmin': gt_data[0,0,i,3],'ymin': gt_data[0,0,i,4],
			'xmax': gt_data[0,0,i,5],'ymax': gt_data[0,0,i,6]})
	return bbox
		

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

def decodeBBox(prior_box,prior_variance,bbox):
	"""
	Adjust a single prior box using the predicted offsets

	Parameters
	----------
	prior_box, prior_variance, bbox: dictionaries with the following keys: 'label','xmin','xmax','ymin','ymax'

	"""
	prior_width = prior_box['xmax'] - prior_box['xmin']
	prior_height = prior_box['ymax'] - prior_box['ymin']
	prior_center_x = (prior_box['xmax'] + prior_box['xmin'])/2.0
	prior_center_y = (prior_box['ymax'] + prior_box['ymin'])/2.0

	decode_bbox_center_x = prior_variance['xmin']*bbox['xmin']*prior_width*prior_center_x
	decode_bbox_center_y = prior_variance['ymin']*bbox['ymin']*prior_height*prior_center_y
	decode_bbox_width = np.exp(prior_variance['xmax']*bbox['xmax'])*prior_width
	decode_bbox_height = np.exp(prior_variance['ymax']*bbox['ymax'])*prior_height

	decode_bbox = {}
	decode_bbox['xmin'] = decode_bbox_center_x - decode_bbox_width/2.0
	decode_bbox['ymin'] = decode_bbox_center_y - decode_bbox_height/2.0
	decode_bbox['xmax'] = decode_bbox_center_x + decode_bbox_width/2.0
	decode_bbox['ymax'] = decode_bbox_center_y + decode_bbox_height/2.0
	return decode_bbox

def decodeBBoxes(prior_bboxes,prior_variances,bboxes):
	"""
	Convert the bounding boxes to take into account the predicted offsets

	Parameters
	----------
	prior_bboxes: list (length num_priors) of dictionaries. Each dictionary contains prior box label and locations ('label','xmin','xmax','ymin','ymax')
	prior_varaiances: list (length num_priors) of dictionaries. Each dictionary contains prior box label and variance ('label','xmin','xmax','ymin','ymax')
	bboxes: list (length num_priors) of dictionaries containing the predicted offsets for the prior box locations ('label','xmin','xmax','ymin','ymax')

	"""
	decode_bboxes = []
	num_bboxes = len(prior_bboxes)
	for i in range(0,num_bboxes):
		decode_bbox_temp = decodeBBox(prior_bboxes[i],prior_variances[i],bboxes[i])
		decode_bboxes.append(decode_bbox_temp)
	return decode_bboxes

#def matchBBOX(gt_bboxes,pred_bboxes,overlap_threshold):

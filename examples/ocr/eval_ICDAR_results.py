# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 15:12:18 2016

@author: marissac
"""
# Use the COCO-Text API to compute the precision and recall metrics on ICDAR data
import sys
sys.path.append("/Users/marissac/Documents/COCOText/coco-text-master")

import coco_text
ct = coco_text.COCO_Text()

imgSize = 300.0

import coco_evaluation
import pickle

icdar_dataset = "icdar-2011"
data_type = "test-textloc-gt"

our_results = coco_text.COCO_Text()

with open('/Users/marissac/caffe/examples/ssd/python/detection_' + icdar_dataset + '_' + data_type + '_anns.pickle','rb') as f1:
    anns_start = pickle.load(f1)
with open('/Users/marissac/caffe/examples/ssd/python/detection_' + icdar_dataset + '_' + data_type + '_imgToAnns.pickle','rb') as f2:
    imgToAnns_start = pickle.load(f2)
with open('/Users/marissac/caffe/examples/ssd/python/detection_' + icdar_dataset + '_gt_' + data_type + '_anns.pickle','rb') as f3:
    gt_anns_start = pickle.load(f3)
with open('/Users/marissac/caffe/examples/ssd/python/detection_' + icdar_dataset + '_gt_' + data_type + '_imgToAnns.pickle','rb') as f4:
    gt_imgToAnns_start = pickle.load(f4)
    
our_results.anns = anns_start
our_results.imgToAnns = imgToAnns_start
    
ct.anns = gt_anns_start
ct.imgToAnns = gt_imgToAnns_start
    
our_results.anns = anns_start
our_results.imgToAnns = imgToAnns_start
    
ct.anns = gt_anns_start
ct.imgToAnns = gt_imgToAnns_start  

imgIdsTotal = imgToAnns_start.keys()

confidence = 0.2
our_results_reduced = coco_evaluation.reduceDetections(our_results, confidence_threshold = confidence)
our_detections = coco_evaluation.getDetections(ct,our_results_reduced, detection_threshold = 0.5)
overallEval = coco_evaluation.evaluateEndToEnd(ct, our_results_reduced, imgIds =  imgIdsTotal, detection_threshold = 0.5)
coco_evaluation.printDetailedResults(ct, our_detections , overallEval, 'yahoo')
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 15:12:18 2016

@author: marissac
"""
# Use the COCO-Text API to compute the precision and recall metrics on ICDAR data

import coco_text
ct = coco_text.COCO_Text()

imgSize = 300.0

import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import coco_evaluation
import pickle

our_results = coco_text.COCO_Text()

with open('/Users/marissac/caffe/examples/ssd/python/detection_icdar-2011_test_anns.pickle','rb') as f1:
    anns_start = pickle.load(f1)
with open('/Users/marissac/caffe/examples/ssd/python/detection_icdar-2011_test_imgToAnns.pickle','rb') as f2:
    imgToAnns_start = pickle.load(f2)
with open('/Users/marissac/caffe/examples/ssd/python/detection_icdar-2011_gt_anns.pickle','rb') as f3:
    gt_anns_start = pickle.load(f3)
with open('/Users/marissac/caffe/examples/ssd/python/detection_icdar-2011_gt_imgToAnns.pickle','rb') as f4:
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

confidence = 0.23
our_results_reduced = coco_evaluation.reduceDetections(our_results, confidence_threshold = confidence)
our_detections = coco_evaluation.getDetections(ct,our_results_reduced, detection_threshold = 0.5)
overallEval = coco_evaluation.evaluateEndToEnd(ct, our_results_reduced, imgIds =  imgIdsTotal, detection_threshold = 0.5)
coco_evaluation.printDetailedResults(ct, our_results_reduced , overallEval, 'yahoo')
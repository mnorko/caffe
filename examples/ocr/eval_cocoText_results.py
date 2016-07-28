# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 15:12:18 2016

@author: marissac
"""
# Compute the precision and recall values for COCO-Text
import sys
sys.path.append("/Users/marissac/Documents/COCOText/coco-text-master")

import coco_text

imgSize = 300.0

import coco_evaluation
import pickle


our_results = coco_text.COCO_Text()


numImg = 20000

with open('/Users/marissac/data/cocoText/results/SSD_300x300/instances/detection_cocoText_anns_' + repr(numImg) + '_legibleSplit.pickle','rb') as f1:
    anns_start = pickle.load(f1)
with open('/Users/marissac/data/cocoText/results/SSD_300x300/instances/detection_cocoText_imgToAnns_' + repr(numImg) + '_legibleSplit.pickle','rb') as f2:
    imgToAnns_start = pickle.load(f2)

ct = coco_text.COCO_Text('/Users/marissac/data/coco/annotations/COCO_Text.json')
our_results.anns = anns_start
our_results.imgToAnns = imgToAnns_start

imgIdsTotal = imgToAnns_start.keys()

confidence = 0.12
our_results_reduced = coco_evaluation.reduceDetections(our_results, confidence_threshold = confidence)
our_detections = coco_evaluation.getDetections(ct,our_results_reduced, detection_threshold = 0.5)
overallEval = coco_evaluation.evaluateEndToEnd(ct, our_results_reduced, imgIds =  imgIdsTotal, detection_threshold = 0.5)
coco_evaluation.printDetailedResults(ct, our_detections , overallEval, 'yahoo')

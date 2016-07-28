# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 13:23:47 2016

@author: marissac
"""
import sys
sys.path.append("/Users/marissac/Documents/COCOText/coco-text-master")

import coco_text
ct = coco_text.COCO_Text()

imgSize = 300.0

import numpy as np
import matplotlib.pyplot as plt
import coco_evaluation
import pickle


our_results = coco_text.COCO_Text()

icdar_dataset = "icdar-2011"
data_type = "test-textloc-gt"

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



leg_eng_mp = ct.getAnnIds(imgIds=[], catIds=[('legibility','legible'),('language','english'),('class','machine printed')], areaRng=[])
leg_eng_hw = ct.getAnnIds(imgIds=[], catIds=[('legibility','legible'),('language','english'),('class','handwritten')], areaRng=[])
leg_mp  = ct.getAnnIds(imgIds=[], catIds=[('legibility','legible'),('class','machine printed')], areaRng=[])
ileg_mp = ct.getAnnIds(imgIds=[], catIds=[('legibility','illegible'),('class','machine printed')], areaRng=[])
leg_hw  = ct.getAnnIds(imgIds=[], catIds=[('legibility','legible'),('class','handwritten')], areaRng=[])
ileg_hw = ct.getAnnIds(imgIds=[], catIds=[('legibility','illegible'),('class','handwritten')], areaRng=[])
leg_ot  = ct.getAnnIds(imgIds=[], catIds=[('legibility','legible'),('class','others')], areaRng=[])
ileg_ot = ct.getAnnIds(imgIds=[], catIds=[('legibility','illegible'),('class','others')], areaRng=[])
    
confidence = np.linspace(0,0.95,40)
recall_total = []
precision_total = []
found_total = []
n_found_total =[]
fp_total = []
f_score_total = []

for i in range(0,len(confidence)):
    our_results_reduced = coco_evaluation.reduceDetections(our_results, confidence_threshold = confidence[i])
    our_detections = coco_evaluation.getDetections(ct,our_results_reduced, detection_threshold = 0.5)
    found = [x['gt_id'] for x in our_detections['true_positives']]
    n_found = [x['gt_id'] for x in our_detections['false_negatives']]
    fp = [x['eval_id'] for x in our_detections['false_positives']]
    t_recall = 100*len(found)*1.0/(len(coco_evaluation.inter(found+n_found, leg_mp+leg_hw+ileg_mp+ileg_hw)))
    #t_recall = 100*len(found)*1.0/(len(found+n_found))
    t_precision = 100*len(found)*1.0/(len(found+fp))
    f_score_temp = 2*t_recall*t_precision/(t_recall+t_precision)
    
    f_score_total.append(f_score_temp)
    recall_total.append(t_recall)
    precision_total.append(t_precision)
    found_total.append(len(found))
    n_found_total.append(len(n_found))
    fp_total.append(len(fp))
    print 'Step ' + repr(i) +'\n'

plt.figure(1)
plt.plot(recall_total,precision_total)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim((0,100))

plt.figure(2)
plt.plot(confidence,f_score_total)
plt.xlabel('Confidence')
plt.ylabel('f-Score')

plt.figure(3)
plt.plot(confidence,precision_total)
plt.xlabel('Confidence')
plt.ylabel('Precision')

plt.figure(4)
plt.plot(confidence,recall_total)
plt.xlabel('Confidence')
plt.ylabel('Recall')
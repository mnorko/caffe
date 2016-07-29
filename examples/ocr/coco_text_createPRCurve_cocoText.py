# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 13:23:47 2016

@author: marissac
"""
import sys
sys.path.append("/Users/marissac/Documents/COCOText/github/coco-text")
import coco_text
ct = coco_text.COCO_Text('/Users/marissac/data/coco/annotations/COCO_Text.json')
imgIds = ct.getImgIds(imgIds=ct.train, 
                    catIds=[('legibility','legible')])
annIds = ct.getAnnIds(imgIds[0:10])
imgInfo = ct.loadImgs(imgIds[0])
imgSize = 300.0

gt_annsScale = ct.loadAnns(annIds)



import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
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


leg_eng_mp = ct.getAnnIds(imgIds=[], catIds=[('legibility','legible'),('language','english'),('class','machine printed')], areaRng=[])
leg_eng_hw = ct.getAnnIds(imgIds=[], catIds=[('legibility','legible'),('language','english'),('class','handwritten')], areaRng=[])
leg_mp  = ct.getAnnIds(imgIds=[], catIds=[('legibility','legible'),('class','machine printed')], areaRng=[])
ileg_mp = ct.getAnnIds(imgIds=[], catIds=[('legibility','illegible'),('class','machine printed')], areaRng=[])
leg_hw  = ct.getAnnIds(imgIds=[], catIds=[('legibility','legible'),('class','handwritten')], areaRng=[])
ileg_hw = ct.getAnnIds(imgIds=[], catIds=[('legibility','illegible'),('class','handwritten')], areaRng=[])
leg_ot  = ct.getAnnIds(imgIds=[], catIds=[('legibility','legible'),('class','others')], areaRng=[])
ileg_ot = ct.getAnnIds(imgIds=[], catIds=[('legibility','illegible'),('class','others')], areaRng=[])
    
confidence = np.linspace(0,0.98,50)
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
    t_precision = 100*len(found)*1.0/(len(found+fp))
    f_score_temp = 2*t_recall*t_precision/(t_recall+t_precision)
    
    f_score_total.append(f_score_temp)
    recall_total.append(t_recall)
    precision_total.append(t_precision)
    found_total.append(len(found))
    n_found_total.append(len(n_found))
    fp_total.append(len(fp))
    print 'Step ' + repr(i) +'\n'
    
max_id = np.argmax(f_score_total)

plt.figure(1)
plt.plot(recall_total,precision_total)
plt.plot(recall_total[max_id],precision_total[max_id],'r8')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim((0,100))
plt.xlim((0,100))
plt.figtext(0.6,0.85,"Recall at Thresh: %.3f" % recall_total[max_id])
plt.figtext(0.6,0.82,"Precision at Thresh: %.3f" % precision_total[max_id])

plt.figure(2)
plt.plot(confidence,f_score_total)
plt.plot(confidence[max_id],f_score_total[max_id],'r8')
plt.xlabel('Confidence')
plt.ylabel('f-Score')
plt.figtext(0.67,0.85,"Threshold: %.3f" % confidence[max_id])
plt.figtext(0.67,0.82,"Max f-score: %.3f" % f_score_total[max_id])

plt.figure(3)
plt.plot(confidence,precision_total)
plt.plot(confidence[max_id],precision_total[max_id],'r8')
plt.xlabel('Confidence')
plt.ylabel('Precision')
plt.figtext(0.20,0.85,"Threshold: %.3f" % confidence[max_id])
plt.figtext(0.20,0.82,"Precision at Thresh: %.3f" % precision_total[max_id])

plt.figure(4)
plt.plot(confidence,recall_total)
plt.plot(confidence[max_id],recall_total[max_id],'r8')
plt.xlabel('Confidence')
plt.ylabel('Recall')
plt.ylim((0,100))
plt.figtext(0.6,0.85,"Threshold: %.3f" % confidence[max_id])
plt.figtext(0.6,0.82,"Recall at Thresh: %.3f" % recall_total[max_id])

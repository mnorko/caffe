# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 14:29:34 2016

@author: marissac
"""

import sys
sys.path.append("/Users/marissac/Documents/COCOText/github/coco-text")

import coco_text

imgSize = 300.0

import numpy as np
import matplotlib.pyplot as plt
import coco_evaluation
import pickle
import statistics as stats

our_results = coco_text.COCO_Text()


numImg = 20000

with open('/Users/marissac/data/cocoText/results/SSD_300x300/instances/detection_cocoText_anns_' + repr(numImg) + '_legible.pickle','rb') as f1:
    anns_start = pickle.load(f1)
with open('/Users/marissac/data/cocoText/results/SSD_300x300/instances/detection_cocoText_imgToAnns_' + repr(numImg) + '_legible.pickle','rb') as f2:
    imgToAnns_start = pickle.load(f2)

ct = coco_text.COCO_Text('/Users/marissac/data/coco/annotations/COCO_Text.json')
our_results.anns = anns_start
our_results.imgToAnns = imgToAnns_start


imgIdsTotal = imgToAnns_start.keys()

del anns_start
del imgToAnns_start

test_run = "hist_plot"
# Options for test run:
#   - bin_behave: plot the statistics for different confidence bins
#   - hist_plot: plot the histograms of the the sizes of bounding boxes for legible and illegible data
#   - gt_stat: Plot ground truth statistics


if test_run == "bin_behave":
    annIds = our_results.getAnnIds(imgIdsTotal)
    anns = our_results.loadAnns(annIds)
    del annIds
    
    bbox_all = [d['bbox'] for d in anns]
    width = [d[2] for d in bbox_all]
    height = [d[3] for d in bbox_all]
    conf = [d['score'] for d in anns]
    
    ratio = np.divide(width,height)
    
    width = np.asarray(width)
    height = np.asarray(height)
    
    num_bins = 40
    
    bins = np.array([0,0.02,0.04,0.06,0.08,0.1,0.12,0.14,0.16,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
    #bins = np.linspace(0,1,num_bins)
    num_bins = len(bins)
    bin_loc = np.digitize(conf,bins)
    ratio_avg = np.zeros((num_bins))
    ratio_var = np.zeros((num_bins))
    width_avg = np.zeros((num_bins))
    width_var = np.zeros((num_bins))
    height_avg = np.zeros((num_bins))
    height_var = np.zeros((num_bins))
    for k in range(0,num_bins):
        idx_use = np.where(bin_loc == (k+1))
        bin_ratio = ratio[idx_use[0]]
        bin_width = width[idx_use[0]]
        bin_height = height[idx_use[0]]
        
        ratio_avg[k] = stats.mean(bin_ratio)
        ratio_var[k] = stats.variance(bin_ratio)
        
        width_avg[k] = stats.mean(bin_width)
        width_var[k] = stats.variance(bin_width)
        
        height_avg[k] = stats.mean(bin_height)
        height_var[k] = stats.variance(bin_height)
        
    plt.figure(0)
    plt.errorbar(bins,ratio_avg,ratio_var)
    plt.xlabel('Confidence')    
    plt.ylabel('Mean Aspect Ratio')
    plt.title('Aspect Ratio vs Confidence')
    
    plt.figure(1)
    #plt.errorbar(bins,height_avg,height_var)
    plt.plot(bins,height_avg)
    plt.xlabel('Confidence')    
    plt.ylabel('Mean Bounding Box Height')
    plt.title('Height vs Confidence')
    
    plt.figure(2)
    #plt.errorbar(bins,width_avg,width_var)
    plt.plot(bins,width_avg)
    plt.xlabel('Confidence')    
    plt.ylabel('Mean Bounding Box Width')
    plt.title('Width vs Confidence')
elif test_run == "hist_plot":
    
    legibility_spec = "illegible"
    class_spec = "machine printed"
    num_hist = 500
    confidence = 0.14
    our_results_reduced = coco_evaluation.reduceDetections(our_results, confidence_threshold = confidence)
    our_detections = coco_evaluation.getDetections(ct,our_results_reduced, detection_threshold = 0.5)
    overallEval = coco_evaluation.evaluateEndToEnd(ct, our_results_reduced, imgIds =  imgIdsTotal, detection_threshold = 0.5)
    coco_evaluation.printDetailedResults(ct, our_detections , overallEval, 'yahoo')
    
    # Get False Negative distribution information
    false_neg_annIds = [d['gt_id'] for d in our_detections['false_negatives']]
    false_neg_annInfo = ct.loadAnns(false_neg_annIds)
    #imgIds_small = [d['bbox'] for d in false_neg_annInfo if d['bbox'][2] <= 10]
    bbox_all = [d['bbox'] for d in false_neg_annInfo if ((d['legibility'] == legibility_spec) & (d['class'] == class_spec))]
    #bbox_all = [d['bbox'] for d in false_neg_annInfo]
    num_fn = len(bbox_all)
    width_false_neg = [d[2] for d in bbox_all]
    [count_fn_width,box_fn_width] = np.histogram(width_false_neg ,num_hist)
    height_false_neg = [d[3] for d in bbox_all]
    [count_fn_height,box_fn_height] = np.histogram(height_false_neg ,num_hist)
    aspect_ratio_false_neg = np.divide(width_false_neg,height_false_neg)
    [count_fn_ratio,box_fn_ratio] = np.histogram(aspect_ratio_false_neg ,num_hist)
    
    # Get True Positive Information
    true_pos_gt_annIds = [d['gt_id'] for d in our_detections['true_positives']]
    true_pos_gt_anns = ct.loadAnns(true_pos_gt_annIds)
    bbox_all = [d['bbox'] for d in true_pos_gt_anns if ((d['legibility'] == legibility_spec) & (d['class'] == class_spec))]
    #bbox_all = [d['bbox'] for d in true_pos_gt_anns]
    num_tp = len(bbox_all)
    width_true_pos = [d[2] for d in bbox_all]
    [count_tp_width,box_tp_width] = np.histogram(width_true_pos ,num_hist)
    height_true_pos = [d[3] for d in bbox_all]
    [count_tp_height,box_tp_height] = np.histogram(height_true_pos ,num_hist)
    aspect_ratio_true_pos = np.divide(width_true_pos,height_true_pos)
    [count_tp_ratio,box_tp_ratio] = np.histogram(aspect_ratio_true_pos ,num_hist)
    
    false_pos_est_annIds = [d['eval_id'] for d in our_detections['false_positives']]
    false_pos_est_anns = our_results_reduced.loadAnns(false_pos_est_annIds)
    num_fp = len(false_pos_est_anns)
    bbox_all = [d['bbox'] for d in false_pos_est_anns]
    width_false_pos_est = [d[2] for d in bbox_all]
    [count_fp_est_width,box_fp_est_width] = np.histogram(width_false_pos_est ,num_hist)
    height_false_pos_est = [d[3] for d in bbox_all]
    [count_fp_est_height,box_fp_est_height] = np.histogram(height_false_pos_est ,num_hist)
    aspect_ratio_false_pos_est = np.divide(width_false_pos_est,height_false_pos_est)
    [count_fp_est_ratio,box_fp_est_ratio] = np.histogram(aspect_ratio_false_pos_est ,num_hist)
    
    plt.figure(0)
    #plt.plot(box_fn_width[1:],count_fn_width*num_tp/num_fn,label='False Neg - Legible Only')
    #plt.plot(box_tp_width[1:],count_tp_width,label='True Pos GT - Legible Only')
    plt.plot(box_fn_width[1:],count_fn_width/float(num_fn),label='False Neg')
    plt.plot(box_tp_width[1:],count_tp_width/float(num_tp),label='True Pos GT')
    #plt.plot(box_fp_est_width[1:],count_fp_est_width/float(num_fp),label='False Pos')
    plt.legend()
    
    plt.title('Histogram of GT Bounding Box Widths - Thresh %.2f' % confidence + " - " + legibility_spec + " - " + class_spec)
    plt.xlim((0,250))
    
    plt.figure(1)
    #plt.plot(box_fn_height[1:],count_fn_height*num_tp/num_fp,label='False Neg - Legible Only')
    #plt.plot(box_tp_height[1:],count_tp_height,label='True Pos GT - Legible Only')
    plt.plot(box_fn_height[1:],count_fn_height/float(num_fn),label='False Neg')
    plt.plot(box_tp_height[1:],count_tp_height/float(num_tp),label='True Pos GT')
    #plt.plot(box_fp_est_height[1:],count_fp_est_height/float(num_fp),label='False Pos')
    plt.legend()
    plt.title('Histogram of GT Bounding Box Heights - Thresh %.2f' % confidence + " - " + legibility_spec + " - " + class_spec)
    plt.xlim((0,250))
    
    plt.figure(2)
    #plt.plot(box_fn_ratio[1:],count_fn_ratio*num_tp/num_fp,label='False Neg - Legible Only')
    #plt.plot(box_tp_ratio[1:],count_tp_ratio,label='True Pos GT - Legible Only')
    plt.plot(box_fn_ratio[1:],count_fn_ratio/float(num_fn),label='False Neg')
    plt.plot(box_tp_ratio[1:],count_tp_ratio/float(num_tp),label='True Pos GT')
    #plt.plot(box_fp_est_ratio[1:],count_fp_est_ratio/float(num_fp),label='False Pos')
    plt.legend()
    plt.title('Histogram of GT Bounding Box Aspect Ratios - Thresh %.2f' % confidence + " - " + legibility_spec + " - " + class_spec)
    plt.xlim((0,12))
    
elif test_run == "gt_stat":
    imgAnns = ct.getAnnIds()
    anns = ct.loadAnns(imgAnns)
    
    legible_use = ["legible","illegible"]
    class_use = ["machine printed","handwritten"]
    
    count = 1
    for k in range(0,len(legible_use)):
        for n in range(0,len(class_use)):
            if class_use[n] == "machine printed":
                numHist = 600
            else:
                numHist = 300
                
            
            bbox_all = [d['bbox'] for d in anns if ((d['legibility'] == legible_use[k]) & (d['class'] == class_use[n]))]
            num_units = float(len(bbox_all))
            width = [d[2] for d in bbox_all]
            [count_width,box_width] = np.histogram(width ,numHist)
            height = [d[3] for d in bbox_all]
            [count_height,box_height] = np.histogram(height ,numHist)
            ratio = np.divide(width,height)
            [count_ratio,box_ratio] = np.histogram(ratio ,numHist)
            area = np.multiply(width,height)
            [count_area,box_area] = np.histogram(area ,numHist)
            
            plt.figure(0)
            #plt.subplot(2,2,count)
            plt.plot(box_width[1:],count_width/num_units,label = legible_use[k] + ", " + class_use[n])            
            
            plt.figure(1)
            #plt.subplot(2,2,count)
            plt.plot(box_height[1:],count_height/num_units,label = legible_use[k] + ", " + class_use[n]) 
            
            plt.figure(2)
            #plt.subplot(2,2,count)
            plt.plot(box_ratio[1:],count_ratio/num_units,label = legible_use[k] + ", " + class_use[n]) 
            
            plt.figure(3)
            plt.plot(box_area[1:],count_area/num_units,label = legible_use[k] + ", " + class_use[n]) 
            
            count = count +1
            
    plt.figure(0)
    plt.legend()
    plt.title('Histogram of Widths')
    plt.xlabel('Width')
    plt.xlim((0,250))
    
    plt.figure(1)
    plt.legend()
    plt.title('Histogram of Heights')
    plt.xlabel('Height')
    plt.xlim((0,250))
    
    plt.figure(2)
    plt.legend()
    plt.title('Histogram of Aspect Ratios')
    plt.xlabel('Aspect Ratio')
    plt.xlim((0,15))
    
    plt.figure(3)
    plt.legend()
    plt.title('Histogram of Area')
    plt.xlabel('Area')
    plt.xlim((0,20000))
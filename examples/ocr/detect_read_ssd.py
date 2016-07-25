# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 14:59:33 2016

@author: marissac
"""
import numpy as np
import caffe
import cv2
import math



def ssd_detect_box(image,caffe_net=None, caffe_transformer=None,confidence_threshold = 0.22,image_resize = 300,multiclass_flag = 1,text_class = 81,text_class2 = 82):
    # Set up network and transform image
    caffe_net.blobs['data'].reshape(1,3,image_resize,image_resize)
    transformed_image = caffe_transformer.preprocess('data',image)
    caffe_net.blobs['data'].data[...] = transformed_image
    
    # Get detection results for image
    caffe_net.forward()
    detections_out = caffe_net.blobs['detection_out'].data
    
    det_label = detections_out[0,0,:,1]
    det_conf = detections_out[0,0,:,2]
    det_xmin = detections_out[0,0,:,3]
    det_ymin = detections_out[0,0,:,4]
    det_xmax = detections_out[0,0,:,5]
    det_ymax = detections_out[0,0,:,6]
    
    if multiclass_flag == 1:
        class_indices = [i for i, label in enumerate(det_label) if ((label == text_class) | (label == text_class2))]
        det_conf = det_conf[class_indices]
        det_xmin = det_xmin[class_indices]
        det_ymin = det_ymin[class_indices]
        det_xmax = det_xmax[class_indices]
        det_ymax = det_ymax[class_indices]


    top_indices = [i for i, conf in enumerate(det_conf) if conf > confidence_threshold]
    
    top_conf = det_conf[top_indices]
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]

    num_detect = top_conf.shape[0]
    
    # Get the detections in the format needed for COCOText detection evaliations
    detection_results = []
    for i in range(0,num_detect):
        top_width = top_xmax[i] - top_xmin[i]
        top_height = top_ymax[i] - top_ymin[i]
        bboxTemp = [top_xmin[i], top_ymin[i],top_width,top_height]
        detection_results.append({'bounding_box':bboxTemp,'label':i,'score':top_conf[i]})
        
    return detection_results
    
def synth_read_words(image,detection_boxes,CAFFE_LABEL_TO_CHAR_MAP,net_synth=None, synth_transformer=None):
    gray_img = np.empty((image.shape[0],image.shape[1],1))
    gray_img_temp = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_img[:,:,0] = gray_img_temp

    num_detections = len(detection_boxes)
    for i in range(0,num_detections):
        bboxTemp = detection_boxes[i]['bounding_box']
        x_min_pix = int(math.floor(bboxTemp[0]*image.shape[1]))
        y_min_pix = int(math.floor(bboxTemp[1]*image.shape[0]))
        x_max_pix = int(math.ceil(bboxTemp[0]*image.shape[1] + bboxTemp[2]*image.shape[1]))
        y_max_pix = int(math.ceil(bboxTemp[1]*image.shape[0] + bboxTemp[3]*image.shape[0]))
        text_img = gray_img[y_min_pix:y_max_pix,x_min_pix:x_max_pix]
        synth_transform_image = synth_transformer.preprocess('data',text_img)
        net_synth.blobs['data'].data[0,0,:,:] = synth_transform_image[0,:,:]
        net_synth.forward()
        output = net_synth.blobs['reshape'].data
        text_out = np.reshape(output,(39,23))
        text_max = np.argmax(text_out, axis=0) 
        output_word = ''
        for j in range(0,23):
            output_word = output_word + CAFFE_LABEL_TO_CHAR_MAP[text_max[j]-1]
            
        detection_boxes[i]['text'] = output_word.strip()
    return detection_boxes
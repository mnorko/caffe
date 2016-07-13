# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 12:59:29 2016

@author: marissac
"""

import numpy as np
import matplotlib.pyplot as plt

# Make sure that caffe is on the python path:
caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
import os
os.chdir(caffe_root)
import sys
sys.path.insert(0, 'python')
#sys.path.append("/Users/marissac/Documents/COCOText/coco-text-master")
sys.path.append("/Users/marissac/Documents/COCOText/github/coco-text")
import caffe
import coco_text
import coco_evaluation
sys.path.append("/Users/marissac/caffe/examples/ocr")
import detect_read_ssd
import skimage.io as io
import cv2
import math
import sys
from google.protobuf import text_format
from caffe.proto import caffe_pb2
import pickle
import time

CAFFE_LABEL_TO_CHAR_MAP = {
    0: '0',
    1: '1',
    2: '2',
    3: '3',
    4: '4',
    5: '5',
    6: '6',
    7: '7',
    8: '8',
    9: '9',
    10: 'a',
    11: 'b',
    12: 'c',
    13: 'd',
    14: 'e',
    15: 'f',
    16: 'g',
    17: 'h',
    18: 'i',
    19: 'j',
    20: 'k',
    21: 'l',
    22: 'm',
    23: 'n',
    24: 'o',
    25: 'p',
    26: 'q',
    27: 'r',
    28: 's',
    29: 't',
    30: 'u',
    31: 'v',
    32: 'w',
    33: 'x',
    34: 'y',
    35: 'z',
    36: ' ',
    37: '\0',
    38: '\0'
}

def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    #print num_labels
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in xrange(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames
    
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

#coco_labelmap_file = '/Users/marissac/caffe/data/coco/labelmap_coco_combo.prototxt'
coco_labelmap_file = '/Users/marissac/caffe/data/coco/labelmap_coco_combo_legible.prototxt'
file = open(coco_labelmap_file, 'r')
coco_labelmap = caffe_pb2.LabelMap()
text_format.Merge(str(file.read()), coco_labelmap)
        
# Define the network to use for text detection
#model_def = '/Users/marissac/caffe/models/VGGNet/cocoText/SSD_300x300/deploy_digitIn_multiclass.prototxt'
model_def = '/Users/marissac/caffe/models/VGGNet/cocoText/SSD_300x300/deploy_digitIn_multiclass_legibleSplit.prototxt'
#model_weights = '/Users/marissac/caffe/examples/ssd/models/VGGNet/cocoText/SSD_300x300/VGG_cocoText_SSD_300x300_multiclass_iter_120000.caffemodel'
#model_weights = '/Users/marissac/caffe/examples/ssd/models/VGGNet/cocoText/SSD_300x300/VGG_cocoText_SSD_300x300_multiclass_corr_iter_204000.caffemodel'
model_weights = '/Users/marissac/caffe/examples/ssd/models/VGGNet/cocoText/SSD_300x300/VGG_cocoText_legibleSplit_SSD_300x300_iter_60000.caffemodel'
net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)
                
# Define the word reader network
synth_model = '/Users/marissac/caffe/examples/ocr/90ksynth/deploy_2.prototxt'
synth_model_weights = '/Users/marissac/caffe/examples/ocr/90ksynth/90ksynth_v2_v00_iter_140000.caffemodel'

net_synth = caffe.Net(synth_model,
                     synth_model_weights,
                     caffe.TEST)
                
# Define the image data transformer
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.array([104,117,123])) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

# Define the data tranformer for the word reader
mean_blob = caffe.proto.caffe_pb2.BlobProto()
data_use = open('/Users/marissac/caffe/examples/ocr/90ksynth/90ksynth_leveldb_mean.binaryproto','rb').read()
mean_blob.ParseFromString(data_use)
mean_arr = np.array( caffe.io.blobproto_to_array(mean_blob) )
mean_val = np.mean(mean_arr)
print mean_val
synth_transformer = caffe.io.Transformer({'data': net_synth.blobs['data'].data.shape})
synth_transformer.set_transpose('data', (2, 0, 1))
synth_transformer.set_mean('data', np.array([mean_val])/256.0)

# Use ground truth data to select images to use
dataDir='/Users/marissac/data/coco'
dataType='train2014'
ct = coco_text.COCO_Text('/Users/marissac/data/coco/annotations/COCO_Text.json')
imgIds = ct.getImgIds(imgIds=ct.val,catIds=[('legibility','legible'),('language','english')]) # Get image IDs for validation images

load_results_flag = 0
# Define the detection COCOText file
detection_cocoText = coco_text.COCO_Text()

if load_results_flag == 1:
    with open('/Users/marissac/caffe/examples/ssd/python/detection_cocoText_anns.pickle','rb') as f1:
        anns_start = pickle.load(f1)
    with open('/Users/marissac/caffe/examples/ssd/python/detection_cocoText_imgToAnns.pickle','rb') as f2:
        imgToAnns_start = pickle.load(f2)
    detection_cocoText.anns = anns_start
    detection_cocoText.imgToAnns = imgToAnns_start
    imgStart = len(imgToAnns_start)
    annStart = len(anns_start)
    imgIdsTotal = imgToAnns_start.keys()
elif load_results_flag == 0:
    detection_cocoText.anns = {}
    annStart = 0
    imgStart = 0
    imgIdsTotal = []

numImgTest = 500
thresh_use = 0.14
detect_time = []
read_time = []
num_detect_total = []
for img_idx in range(imgStart,imgStart+numImgTest):
    
    img = ct.loadImgs(imgIds[img_idx])[0] # Select an image to use
    imgIdsTotal.append(imgIds[img_idx])
    #imgNumTest = 258869
    #img = ct.loadImgs(imgNumTest)[0]
    image = caffe.io.load_image('%s/images/%s/%s'%(dataDir,dataType,img['file_name']))
    
    detection_read = []
    
    detect_tic = time.time()
    
    #detection_read= detect_read_ssd.ssd_detect_box(image,net, transformer,confidence_threshold = thresh_use,image_resize = 300,multiclass_flag = 1,text_class = 81)
    detection_read= ssd_detect_box(image,net, transformer,confidence_threshold = thresh_use,image_resize = 300,multiclass_flag = 1,text_class = 81,text_class2 = 82)
    #= ssd_detect_box(image,net, transformer,confidence_threshold = thresh_use,image_resize = 300,multiclass_flag = 1,text_class = 81)

    detect_toc = time.time()
    detect_time.append(detect_toc-detect_tic)
    # Finish defining fields in detection cocoText info
    imgUse = img['id']
    num_detect = len(detection_read)
    detection_cocoText.imgToAnns[imgUse] = list(np.linspace(annStart,annStart+num_detect-1,num_detect))
    num_detect_total.append(num_detect)
    read_tic = time.time()
    detection_read = detect_read_ssd.synth_read_words(image,detection_read,CAFFE_LABEL_TO_CHAR_MAP,net_synth, synth_transformer)
    #detection_read = synth_read_words(image,detection_read,CAFFE_LABEL_TO_CHAR_MAP,net_synth, synth_transformer)
    read_toc = time.time()
    read_time.append(read_toc-read_tic)
    for i in range(0,num_detect):
        bbox_temp = detection_read[i]['bounding_box']
        bbox_use = [bbox_temp[0]*image.shape[1], bbox_temp[1]*image.shape[0], bbox_temp[2]*image.shape[1], bbox_temp[3]*image.shape[0]]
        area_use = bbox_use[2]*bbox_use[3]
        detection_cocoText.anns[annStart+i] = {'area':area_use,'bbox':bbox_use,
        'category_id':1,'id':annStart+i,'image_id':imgUse,'score':detection_read[i]['score'],'utf8_string':detection_read[i]['text']}

    annStart = annStart+num_detect
    if img_idx % 10 == 0:
        print 'Step ' + repr(img_idx) +'\n'
# Find the boxes that match up with the ground truth information
detection_img = coco_evaluation.getDetections(ct,detection_cocoText,imgIds = imgIdsTotal,detection_threshold = 0.5)
overallEval = coco_evaluation.evaluateEndToEnd(ct, detection_cocoText, imgIds =  imgIdsTotal, detection_threshold = 0.5)
coco_evaluation.printDetailedResults(ct, detection_img , overallEval, 'yahoo')

with open('detection_cocoText_anns_' + repr(imgStart+numImgTest) + '_legible.pickle','wb') as f:
    pickle.dump(detection_cocoText.anns,f)
    
with open('detection_cocoText_imgToAnns_' + repr(imgStart+numImgTest) + '_legible.pickle','wb') as f:
    pickle.dump(detection_cocoText.imgToAnns,f)
## Check how we're doing compared to ground truth
#true_detections = detection_img['true_positives']
#num_true_detect = len(true_detections)
#for i in range(0,num_true_detect):
#
#    # Check if ground truth is legible and english
#    gtAnnInfo = ct.loadAnns(true_detections[i]['gt_id'])
#    if (gtAnnInfo[0]['language'] == 'english') & (gtAnnInfo[0]['legibility'] == 'legible'):
#        # Get the bounding box associated with the detection
#        detect_id = true_detections[i]['eval_id']  
#        detection_cocoText.anns[detect_id]['utf8_string'] = output_word.strip()

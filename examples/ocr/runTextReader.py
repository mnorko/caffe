# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 12:59:29 2016

@author: marissac
"""

import numpy as np
import matplotlib.pyplot as plt

# Make sure that caffe is on the python path:
caffe_root = '../../'  
import os
os.chdir(caffe_root)
import sys
sys.path.insert(0, 'python')
sys.path.append("/Users/marissac/Documents/COCOText/github/coco-text")
import caffe
import coco_text
import coco_evaluation
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

coco_labelmap_file = '/Users/marissac/caffe/data/coco/labelmap_coco_combo_legible.prototxt'
file = open(coco_labelmap_file, 'r')
coco_labelmap = caffe_pb2.LabelMap()
text_format.Merge(str(file.read()), coco_labelmap)
        
# Define the network to use for text detection
model_def = '/Users/marissac/caffe/models/VGGNet/cocoText/SSD_300x300/deploy_digitIn_multiclass_legibleSplit.prototxt'
model_weights = '/Users/marissac/caffe/examples/ssd/models/VGGNet/cocoText/SSD_300x300/VGG_cocoText_legibleSplit_SSD_300x300_iter_244000.caffemodel'
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
    image = caffe.io.load_image('%s/images/%s/%s'%(dataDir,dataType,img['file_name']))
    
    detection_read = []
    
    detect_tic = time.time()
    
    detection_read= detect_read_ssd.ssd_detect_box(image,net, transformer,confidence_threshold = thresh_use,image_resize = 300,multiclass_flag = 1,text_class = 81,text_class2 = 82)

    detect_toc = time.time()
    detect_time.append(detect_toc-detect_tic)

    # Finish defining fields in detection cocoText info
    imgUse = img['id']
    num_detect = len(detection_read)
    detection_cocoText.imgToAnns[imgUse] = list(np.linspace(annStart,annStart+num_detect-1,num_detect))
    num_detect_total.append(num_detect)
    read_tic = time.time()
    detection_read = detect_read_ssd.synth_read_words(image,detection_read,CAFFE_LABEL_TO_CHAR_MAP,net_synth, synth_transformer)
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

# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 12:59:29 2016

@author: marissac
"""

import numpy as np

# Make sure that caffe is on the python path:
caffe_root = '../../'  
import os
os.chdir(caffe_root)
import sys
sys.path.insert(0, 'python')

import caffe
import math
import sys
from google.protobuf import text_format
from caffe.proto import caffe_pb2
import pickle
import time
import lmdb
from PIL import Image
from StringIO import StringIO

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


def synth_read_words(image,detection_boxes,CAFFE_LABEL_TO_CHAR_MAP,net_synth=None, synth_transformer=None):
    gray_img = np.empty((image.shape[0],image.shape[1],1))
    gray_img_temp = 0.299*image[:,:,0] + 0.587*image[:,:,1] + 0.114*image[:,:,2]
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

coco_labelmap_file = '/home/marissac/ssd/caffe/data/coco/labelmap_coco_combo_legible.prototxt'
file = open(coco_labelmap_file, 'r')
coco_labelmap = caffe_pb2.LabelMap()
text_format.Merge(str(file.read()), coco_labelmap)

caffe.set_mode_gpu()        
# Define the network to use for text detection
model_def = '/home/marissac/ssd/caffe/examples/ssd/models/VGGNet/cocoText/SSD_300x300/test_multiclass_ssh_legibleSplit.prototxt'
model_weights = '/home/marissac/ssd/caffe/examples/ssd/models/VGGNet/cocoText/SSD_300x300/VGG_cocoText_legibleSplit_SSD_300x300_iter_244000.caffemodel'
net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)
                
# Define the word reader network
synth_model = '/home/marissac/ssd/caffe/examples/ocr/90ksynth/deploy_2.prototxt'
synth_model_weights = '/home/marissac/ssd/caffe/examples/ocr/90ksynth/90ksynth_v2_v00_iter_140000.caffemodel'

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
data_use = open('/home/marissac/ssd/caffe/examples/ocr/90ksynth/90ksynth_leveldb_mean.binaryproto','rb').read()
mean_blob.ParseFromString(data_use)
mean_arr = np.array( caffe.io.blobproto_to_array(mean_blob) )
mean_val = np.mean(mean_arr)
print mean_val
synth_transformer = caffe.io.Transformer({'data': net_synth.blobs['data'].data.shape})
synth_transformer.set_transpose('data', (2, 0, 1))
synth_transformer.set_mean('data', np.array([mean_val])/255.0)

name_size_file = '/home/marissac/ssd/caffe/data/coco/combo_val_name_size.txt'
nameFile = open(name_size_file,'r')
nameFileAll = nameFile.readlines()

load_results_flag = 0


annStart = 0
imgStart = 0
imgIdsTotal = []

numImgTest = 20000 # Number of test images to analyze
thresh_use = 0.0 # Threshold for detections
text_class =81 # legible text class
text_class2 = 82 # illegible text class
multiclass_flag = 1 # Flag to indicate whether the dataset has more than one class
test_label = "legibleSplit" # Label for the output file

detect_time = []
read_time = []
num_detect_total = []
imgToAnns = {}
anns = {}

lmdb_env = lmdb.open('/home/marissac/data/coco/coco_combo_legible_val2014_final_lmdb')
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()
datum = caffe.proto.caffe_pb2.AnnotatedDatum()

for img_idx in range(imgStart,imgStart+numImgTest):
    detection_read = []
    
    detect_tic = time.time()

    # Get detection results for image
    net.forward()
    detections_out = net.blobs['detection_out'].data
    
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


    top_indices = [i for i, conf in enumerate(det_conf) if conf > thresh_use]
    
    top_conf = det_conf[top_indices]
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]

    num_detect = top_conf.shape[0]
    
    # Get the detections in the format needed for COCOText detection evaliations
    detection_read = []
    for i in range(0,num_detect):
        top_width = top_xmax[i] - top_xmin[i]
        top_height = top_ymax[i] - top_ymin[i]
        bboxTemp = [top_xmin[i], top_ymin[i],top_width,top_height]
        detection_read.append({'bounding_box':bboxTemp,'label':i,'score':top_conf[i]})
            
    detect_toc = time.time()
    detect_time.append(detect_toc-detect_tic)
    # Finish defining fields in detection cocoText info
    file_name_line = nameFileAll[img_idx]
    imgNum, height, width = file_name_line.split(" ")
    imgUse = int(imgNum)
    height_use = int(height)
    width_use = int(width)
    num_detect = len(detection_read)
    # Define the imgToAnns which gives a number to each annotation - required for COCO-Text API
    imgToAnns[imgUse] = list(np.linspace(annStart,annStart+num_detect-1,num_detect))
    num_detect_total.append(num_detect)
    read_tic = time.time()
    
    # Read image from the lmdb 
    lmdb_cursor.next_nodup()
    value = lmdb_cursor.value()
    datum.ParseFromString(value)
    jpg = datum.datum.data
    image = np.array(Image.open(StringIO(jpg)))
    image = image/255.0
    
    # Read word
    detection_read = synth_read_words(image,detection_read,CAFFE_LABEL_TO_CHAR_MAP,net_synth, synth_transformer)
    read_toc = time.time()
    read_time.append(read_toc-read_tic)
    for i in range(0,num_detect):
        bbox_temp = detection_read[i]['bounding_box']
        bbox_use = [bbox_temp[0]*width_use, bbox_temp[1]*height_use, bbox_temp[2]*width_use, bbox_temp[3]*height_use]
        area_use = bbox_use[2]*bbox_use[3]
        anns[annStart+i] = {'area':area_use,'bbox':bbox_use,
        'category_id':1,'id':annStart+i,'image_id':imgUse,'score':detection_read[i]['score'],'utf8_string':detection_read[i]['text']}
        
    annStart = annStart+num_detect
    # Save snapshots along the way
    if img_idx % 10 == 0:
        print 'Step ' + repr(img_idx) +'\n'
    if img_idx % 10000 == 0:
        with open('/home/marissac/data/cocoText/instances/detection_cocoText_anns_' + repr(imgStart+img_idx) + '_' + test_label +'.pickle','wb') as f:
            pickle.dump(anns,f)
    
        with open('/home/marissac/data/cocoText/instances/detection_cocoText_imgToAnns_' + repr(imgStart+img_idx) + '_' + test_label + '.pickle','wb') as f:
            pickle.dump(imgToAnns,f)

with open('/home/marissac/data/cocoText/instances/detection_cocoText_anns_' + repr(imgStart+numImgTest) + '_' + test_label + '.pickle','wb') as f:
    pickle.dump(anns,f)
    
with open('/home/marissac/data/cocoText/instances/detection_cocoText_imgToAnns_' + repr(imgStart+numImgTest) + '_' + test_label + 'pickle','wb') as f:
    pickle.dump(imgToAnns,f)


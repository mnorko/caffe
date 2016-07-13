# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 11:20:20 2016

@author: marissac
"""

import numpy as np
import matplotlib.pyplot as plt
import lmdb
from PIL import Image
from StringIO import StringIO
import sys
sys.path.append("/Users/marissac/caffe/examples/pycaffe/layers")
sys.path.append("/Users/marissac/Documents/COCOText/coco-text-master")

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Make sure that caffe is on the python path:
caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
import os
os.chdir(caffe_root)

import caffe
import coco_text
import coco_evaluation
from google.protobuf import text_format
from caffe.proto import caffe_pb2
import cv2

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

model_def = '/Users/marissac/caffe/examples/ocr/90ksynth/deploy_ocr_spatial_transform.prototxt'
model_weights = '/Users/marissac/caffe/examples/ocr/90ksynth/90ksynth_v2_v00_iter_140000.caffemodel'


net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

mean_blob = caffe.proto.caffe_pb2.BlobProto()
data_use = open('/Users/marissac/caffe/examples/ocr/90ksynth/90ksynth_leveldb_mean.binaryproto','rb').read()
mean_blob.ParseFromString(data_use)
mean_arr = np.array( caffe.io.blobproto_to_array(mean_blob) )
mean_val = np.mean(mean_arr)
print mean_val
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.array([mean_val])/256.0)


ct = coco_text.COCO_Text('/Users/marissac/data/coco/annotations/COCO_Text.json')
imgNumTest = 151934
img = ct.loadImgs(imgNumTest)[0]

dataDir='/Users/marissac/data/coco'
dataType='train2014'



image = caffe.io.load_image('%s/images/%s/%s'%(dataDir,dataType,img['file_name']))
annIds = ct.getAnnIds(imgIds=img['id'])
anns = ct.loadAnns(annIds)                
#net_test = caffe.Net(model_test,caffe.TEST)
#net_test.forward(

gray_img = np.empty((image.shape[0],image.shape[1],1))
gray_img_temp = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
gray_img[:,:,0] = gray_img_temp

synth_transform_image = transformer.preprocess('data',gray_img)
net.blobs['data'].data[0,0,:,:] = synth_transform_image[0,:,:]
        

# Create an image with an MNIST digit had a specified location
width_input = 640.0
height_input = 480.0

bbox_all = [d['bbox'] for d in anns]
num_bboxes = len(bbox_all)
theta = np.zeros((num_bboxes,6))
for k in range(0,num_bboxes):
    bbox_size_x = bbox_all[k][2]
    bbox_size_y = bbox_all[k][3]
    scale_x = bbox_size_x/width_input
    scale_y = bbox_size_y/height_input
    loc_digit_x = bbox_all[k][0]
    loc_digit_y = bbox_all[k][1]
    bbox_x = 2*(-(width_input-bbox_size_x)/2 + loc_digit_x)/width_input
    bbox_y = 2*(-(height_input-bbox_size_y)/2 + loc_digit_y)/height_input
    theta[k,:] = np.array([scale_x,0,bbox_x,0,scale_y,bbox_y])

net.blobs['theta'].data[...] = theta
net.forward()    

output = net.blobs['reshape'].data
words_final = []
for k in range(0,num_bboxes):
    text_out = np.reshape(output[k,:,:,:],(39,23))
    text_max = np.argmax(text_out, axis=0) 
    output_word = ''
    for j in range(0,23):
        output_word = output_word + CAFFE_LABEL_TO_CHAR_MAP[text_max[j]-1]
        
    words_final.append(output_word)
    img_transform = net.blobs['data_transform'].data[k,0,:,:]
    plt.figure(k)
    plt.imshow(img_transform)
#imgTemp = np.zeros((height_input,width_input))
#loc_digit = np.array([[10,20],[52,20]]) # box x location, box y location
#num_detections = 2
#for loc_idx in range(0,num_detections):
#    imgTemp[loc_digit[loc_idx,1]:loc_digit[loc_idx,1]+28,loc_digit[loc_idx,0]:loc_digit[loc_idx,0]+28] =transformed_image[0,:,:]
##imgTemp = transformed_image[0,:,:]
#imgUse = np.expand_dims(imgTemp,0)
#
## Find digit
#bbox_size_x = 28.0
#bbox_size_y = 28.0
#scale_x = bbox_size_x/width_input
#scale_y = bbox_size_y/height_input
#theta = np.zeros((num_detections,6))
#for loc_idx in range(0,num_detections):
#    bbox_x = 2*(-(width_input-bbox_size_x)/2 + loc_digit[loc_idx,0])/width_input
#    bbox_y = 2*(-(height_input-bbox_size_y)/2 + loc_digit[loc_idx,1])/height_input
#    theta[loc_idx,:] = np.array([scale_x,0,bbox_x,0,scale_y,bbox_y])
##theta = np.array([1,0,0,0,1,0])
#
#net.blobs['theta'].data[...] = theta
#lmdb_cursor.next_nodup()
#value = lmdb_cursor.value()
#datum.ParseFromString(value)
#label = datum.label
#data = caffe.io.datum_to_array(datum)
#test= 1
#    
#net.forward()   
##transformed_image = transformer.preprocess('data',image)
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 11:08:10 2016

@author: marissac
"""

import numpy as np
import matplotlib.pyplot as plt

import json
from PIL import Image
import glob
caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
import os
os.chdir(caffe_root)
import sys
sys.path.insert(0, 'python')
sys.path.append("/Users/marissac/Documents/COCOText/coco-text-master")
import caffe
import coco_text
import coco_evaluation

from google.protobuf import text_format
from caffe.proto import caffe_pb2

# load PASCAL COCO labels
coco_labelmap_file = '/Users/marissac/caffe/data/coco/labelmap_cocoText.prototxt'
file = open(coco_labelmap_file, 'r')
coco_labelmap = caffe_pb2.LabelMap()
text_format.Merge(str(file.read()), coco_labelmap)

def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    print num_labels
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

icdar_dataset = "icdar-2011"
data_type = "test-textloc-gt"

anno_dir = "/Users/marissac/data/ICDAR/" + icdar_dataset + "/" + data_type + "/"
out_dir = "/Users/marissac/data/ICDAR/" + icdar_dataset + "/" + data_type + "-output"

fileNames = glob.glob(anno_dir + '*.jpg')
numFiles = len(fileNames)



model_def = '/Users/marissac/caffe/models/VGGNet/cocoText/SSD_300x300/deploy_selectDigit.prototxt'
model_weights = '/Users/marissac/caffe/examples/ssd/models/VGGNet/cocoText/SSD_300x300/VGG_cocoText_SSD_300x300_iter_140000.caffemodel'
net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.array([104,117,123])) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

imgSize = 300

ct = coco_text.COCO_Text()

# Select a random file

file_id_use = np.random.randint(0,numFiles)
fileNameTemp = fileNames[file_id_use]
# Find the image id from the filename
fileNameTemp = fileNameTemp.rstrip('.jpg')
fileNameTemp = fileNameTemp[len(anno_dir):]
img_id = int(fileNameTemp)
img_file_name = repr(img_id) + '.jpg'

image = caffe.io.load_image('%s/%s'%(anno_dir,img_file_name))

# Load annotations
name = icdar_dataset + "_" + repr(img_id)  
anno_file = "{}/{}.json".format(out_dir, name)
json_data = open(anno_file)
anno_data = json.load(json_data)

anns = anno_data['annotation']
ct.anns = anns
ct.imgToAnns[img_id] = list(np.linspace(0,len(anns)-1,len(anns)))

image_resize = 300
net.blobs['data'].reshape(1,3,image_resize,image_resize)

transformed_image = transformer.preprocess('data',image)
net.blobs['data'].data[...] = transformed_image
#net.blobs['data'].data[...] = test_image
# Forward pass.
net.forward()

detections = net.blobs['detection_out'].data

# Parse the outputs.
det_label = detections[0,0,:,1]
det_conf = detections[0,0,:,2]
det_xmin = detections[0,0,:,3]
det_ymin = detections[0,0,:,4]
det_xmax = detections[0,0,:,5]
det_ymax = detections[0,0,:,6]


# Get detections with confidence higher than 0.6.
top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.23]

top_conf = det_conf[top_indices]
top_label_indices = det_label[top_indices].tolist()
top_labels = get_labelname(coco_labelmap, top_label_indices)
top_xmin = det_xmin[top_indices]
top_ymin = det_ymin[top_indices]
top_xmax = det_xmax[top_indices]
top_ymax = det_ymax[top_indices]

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
plt.figure()
plt.imshow(image)
ct.showAnns(anns)

num_detect = top_conf.shape[0]
detections = coco_text.COCO_Text()
detections.imgToAnns[img_id] = list(np.linspace(0,int(num_detect-1),int(num_detect)))
[int(j) for j in detections.imgToAnns[img_id]]
print detections.imgToAnns[img_id]
print img_id
detections.anns = {}
for i in range(0,num_detect):
    xmin_pix = top_xmin[i] * image.shape[1]
    ymin_pix = top_ymin[i] * image.shape[0]
    width_pix = top_xmax[i] * image.shape[1] - xmin_pix
    height_pix = top_ymax[i] * image.shape[0] - ymin_pix
    area_pix = width_pix*height_pix
    category_id = 1
    id_use = i
    image_id = img_id
    score_use = top_conf[i]
    detections.anns[i] = {'area':area_pix,'bbox':[xmin_pix,ymin_pix,width_pix,height_pix],'category_id':category_id,
                          'id':id_use,'image_id':image_id,'score':score_use}

#our_results_reduced = coco_evaluation.reduceDetections(detections, confidence_threshold = thresh_use)
detection_img = coco_evaluation.getDetections(ct,detections,imgIds = [img_id],detection_threshold = 0.5)

# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 13:26:17 2016

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
import skimage.io as io
import cv2
import math
import sys
from google.protobuf import text_format
from caffe.proto import caffe_pb2
import pickle
import glob
import json


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
        det_label = det_label[class_indices]
        det_xmin = det_xmin[class_indices]
        det_ymin = det_ymin[class_indices]
        det_xmax = det_xmax[class_indices]
        det_ymax = det_ymax[class_indices]


    top_indices = [i for i, conf in enumerate(det_conf) if conf > confidence_threshold]
    
    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].tolist()
    top_labels = get_labelname(coco_labelmap, top_label_indices)
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]

    num_detect = top_conf.shape[0]
    
    # Get the detections in the format needed for COCOText detection evaliations
    detection_results = []
    for i in range(0,num_detect):
        xmin_pix = top_xmin[i] * image.shape[1]
        ymin_pix = top_ymin[i] * image.shape[0]
        width_pix = top_xmax[i] * image.shape[1] - xmin_pix
        height_pix = top_ymax[i] * image.shape[0] - ymin_pix
        top_width = top_xmax[i] - top_xmin[i]
        top_height = top_ymax[i] - top_ymin[i]
        bboxTemp = [top_xmin[i], top_ymin[i],top_width,top_height]
        area = width_pix*height_pix
        detection_results.append({'bounding_box':bboxTemp,'label':i,'score':top_conf[i],'area':area})
        
    return detection_results
    
def synth_read_words(image,detection_boxes,CAFFE_LABEL_TO_CHAR_MAP,net_synth=None, synth_transformer=None):
    img_shape = image.shape
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

icdar_dataset = "icdar-2011"
data_type = "test-textloc-gt"

anno_dir = "/Users/marissac/data/ICDAR/" + icdar_dataset + "/" + data_type + "/"
out_dir = "/Users/marissac/data/ICDAR/" + icdar_dataset + "/" + data_type + "-output"

fileNames = glob.glob(anno_dir + '*.jpg')
numFiles = len(fileNames)


ct = coco_text.COCO_Text()
imgIds = ct.getImgIds(imgIds=ct.val) # Get image IDs for validation images

# Define the detection COCOText file
detection_cocoText = coco_text.COCO_Text()
detection_cocoText.anns = {}
annStart = 0
gtAnnStart = 0
imgStart = 0
imgIdsTotal = []

thresh_use = 0.0

for img_idx in range(0,numFiles):
    
    fileNameTemp = fileNames[img_idx]
    # Find the image id from the filename
    fileNameTemp = fileNameTemp.rstrip('.jpg')
    fileNameTemp = fileNameTemp[len(anno_dir):]
    img_id = int(fileNameTemp)
    img_file_name = repr(img_id) + '.jpg'

    imgIdsTotal.append(img_id)
    image = caffe.io.load_image('%s/%s'%(anno_dir,img_file_name))
    
    
    name = icdar_dataset + "_" + repr(img_id)  
    anno_file = "{}/{}.json".format(out_dir, name)
    json_data = open(anno_file)
    anno_data = json.load(json_data)
    
    anns = anno_data['annotation']
    numGT = len(anns)
    ct.imgToAnns[img_id] = list(np.linspace(gtAnnStart,gtAnnStart+numGT-1,numGT))
    for i in range(0,numGT):
        anns[i]['class'] = 'machine printed'
        anns[i]['legibility'] = 'legible'
        anns[i]['language'] = 'english'
        ct.anns[ct.imgToAnns[img_id][i]] = anns[i]

    detection_read = []
    
    detection_read= ssd_detect_box(image,net, transformer,confidence_threshold = thresh_use,image_resize = 300,multiclass_flag = 1,text_class = 81,text_class2 = 85)
    
    
    # Finish defining fields in detection cocoText info
    imgUse = img_id
    num_detect = len(detection_read)
    detection_cocoText.imgToAnns[imgUse] = list(np.linspace(annStart,annStart+num_detect-1,num_detect))

    detection_read = synth_read_words(image,detection_read,CAFFE_LABEL_TO_CHAR_MAP,net_synth, synth_transformer)
    
    for i in range(0,num_detect):
        bbox_temp = detection_read[i]['bounding_box']
        bbox_use = [bbox_temp[0]*image.shape[1], bbox_temp[1]*image.shape[0], bbox_temp[2]*image.shape[1], bbox_temp[3]*image.shape[0]]
        detection_cocoText.anns[annStart+i] = {'area':detection_read[i]['score'],'bbox':bbox_use,
        'category_id':1,'id':annStart+i,'image_id':imgUse,'score':detection_read[i]['score'],'utf8_string':detection_read[i]['text']}

    annStart = annStart+num_detect
    gtAnnStart = gtAnnStart + numGT
    if img_idx % 10 == 0:
        print 'Step ' + repr(img_idx) +'\n'
# Find the boxes that match up with the ground truth information
detection_img = coco_evaluation.getDetections(ct,detection_cocoText,imgIds = imgIdsTotal,detection_threshold = 0.5)
overallEval = coco_evaluation.evaluateEndToEnd(ct, detection_cocoText, imgIds =  imgIdsTotal, detection_threshold = 0.5)
coco_evaluation.printDetailedResults(ct, detection_img , overallEval, 'yahoo')

with open('detection_' + icdar_dataset + '_' + data_type + '_legOnly_anns.pickle','wb') as f:
    pickle.dump(detection_cocoText.anns,f)
    
with open('detection_' + icdar_dataset + '_' + data_type + '_legOnly_imgToAnns.pickle','wb') as f:
    pickle.dump(detection_cocoText.imgToAnns,f)
    
with open('detection_' + icdar_dataset + '_gt_' + data_type + '_legOnly_anns.pickle','wb') as f:
    pickle.dump(ct.anns,f)
    
with open('detection_' + icdar_dataset + '_gt_' + data_type + '_legOnly_imgToAnns.pickle','wb') as f:
    pickle.dump(ct.imgToAnns,f)


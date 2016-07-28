# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 12:12:09 2016

@author: marissac
"""

# Create cocoText instance json file that can be used for finding AP using the COCO API
import sys
sys.path.insert(0, 'python')
sys.path.append("/Users/marissac/Documents/COCOText/coco-text-master")
sys.path.append("/Users/marissac/Code/github/simplejson")
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import coco_evaluation
import pickle
import os
import json
import simplejson
from decimal import Decimal

# Load in the pickle files

numImg = 20000

with open('/Users/marissac/data/cocoText/results/SSD_300x300/instances/detection_cocoText_anns_' + repr(numImg) + '_legible.pickle','rb') as f1:
    anns_start = pickle.load(f1)
with open('/Users/marissac/data/cocoText/results/SSD_300x300/instances/detection_cocoText_imgToAnns_' + repr(numImg) + '_legible.pickle','rb') as f2:
    imgToAnns_start = pickle.load(f2)

numAnns = len(anns_start)

anno_img = []
for i in range(0,numAnns):
    bbox_use = [float(anns_start[i]['bbox'][0]), float(anns_start[i]['bbox'][1]), float(anns_start[i]['bbox'][2]), float(anns_start[i]['bbox'][3])]
    area = bbox_use[2]*bbox_use[3]
    id_use = anns_start[i]['id']
    img_id = anns_start[i]['image_id']
    score = float(anns_start[i]['score'])
    anno_img.append({"area":area,"bbox":bbox_use,"category_id":1,"id": id_use,"image_id":img_id,"iscrowd":0,"score":score})

categories = {"id":1,"name": 'text',"supercategory":'text'}

img_anno = anno_img


out_dir = "/Users/marissac/caffe/examples/ssd/python/instances"

if not os.path.exists(out_dir):
        os.makedirs(out_dir )       
        
name = 'coco_text_instances_' + repr(numImg) 
anno_file = "{}/{}.json".format(out_dir, name)
with open(anno_file, "w") as f:
    s = json.dumps(img_anno, sort_keys=True, indent=2, ensure_ascii=False).encode('utf8')
    f.write(s)
    
f.close()

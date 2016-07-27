# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 10:48:08 2016

@author: marissac
"""

import argparse
from collections import OrderedDict
import json
import os
from pprint import pprint
import sys
sys.path.append('..')

sys.path.append("/Users/marissac/Documents/COCOText/coco-text-master")

from coco_text import COCO_Text
from pycocotools.coco import COCO

# This code creates json files for the combination of COCOText and COCO datasets

annofile = '/Users/marissac/data/coco/annotations/COCO_Text.json'
coco_annofile = '/Users/marissac/data/coco/annotations/instances_train2014.json'
out_dir_train = '/Users/marissac/data/coco/annotations/combo_legibleOnly_train2014'
out_dir_val = '/Users/marissac/data/coco/annotations/combo_legibleOnly_val2014'
imgset_file_train = '/Users/marissac/data/coco/ImageSets/combo_legibleOnly_train2014_test.txt'
imgset_file_val = '/Users/marissac/data/coco/ImageSets/combo_legibleOnly_val2014_test.txt'



if not os.path.exists(annofile):
    print "{} does not exist!".format(annofile)
    sys.exit()


if out_dir_train:
    if not os.path.exists(out_dir_train):
        os.makedirs(out_dir_train)
if imgset_file_train:
    imgset_dir = os.path.dirname(imgset_file_train)
    if not os.path.exists(imgset_dir):
        os.makedirs(imgset_dir)
if out_dir_val:
    if not os.path.exists(out_dir_val):
        os.makedirs(out_dir_val)
if imgset_file_val:
    imgset_dir = os.path.dirname(imgset_file_val)
    if not os.path.exists(imgset_dir):
        os.makedirs(imgset_dir)

# initialize COCO api.
coco_text = COCO_Text(annofile)

# Get all the image information from COCO_Text
img_ids = coco_text.getImgIds()
img_names_train = []
img_names_val = []
for img_id in img_ids:
    # get image info
    img = coco_text.loadImgs(img_id)
    file_name = img[0]["file_name"]
    setLabel = img[0]["set"]
    name = os.path.splitext(file_name)[0]

    if setLabel == "train":
        out_dir = out_dir_train
    else:
        out_dir = out_dir_val
    if out_dir:
        # get annotation info
        anno_ids = coco_text.getAnnIds(imgIds=img_id)
        anno = coco_text.loadAnns(anno_ids)
        
        anno_total = []
        for d in anno:
            if (d['legibility'] == 'legible') & (d['language'] == 'english'):
                if len(d['utf8_string']) < 24:
                    d['category_id'] = 91 # text
                    anno_total.append(d)
            
        
        
        # save annotation to file
        img_anno = dict()
        img_anno["image"] = img[0]
        img_anno["annotation"] = anno_total
        anno_file = "{}/{}.json".format(out_dir, name)
        with open(anno_file, "w") as f:
            try:
                # json.dump(img_anno, f, sort_keys=True, indent=2, ensure_ascii=False)
                s = json.dumps(img_anno, sort_keys=True, indent=2, ensure_ascii=False).encode('utf8')
                f.write(s)
            except:
                import IPython ; IPython.embed()
    if setLabel == "train":
        if imgset_file_train:
            img_names_train.append(name)        
    else:
        if imgset_file_val:
            img_names_val.append(name)
            
if img_names_train:
    img_names_train.sort()
    with open(imgset_file_train, "w") as f:
        f.write("\n".join(img_names_train))
        
if img_names_val:
    img_names_val.sort()
    with open(imgset_file_val, "w") as f:
        f.write("\n".join(img_names_val))
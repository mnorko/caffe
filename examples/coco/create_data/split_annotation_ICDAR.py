# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 14:54:25 2016

@author: marissac
"""
import json
from PIL import Image
import glob
import os


icdar_dataset = "icdar-2011"
data_type = "train-textloc"

anno_dir = "/Users/marissac/data/ICDAR/" + icdar_dataset + "/" + data_type + "/"
out_dir = "/Users/marissac/data/ICDAR/" + icdar_dataset + "/" + data_type + "-output"

if not os.path.exists(out_dir):
        os.makedirs(out_dir )

fileNames = glob.glob(anno_dir + '*.jpg')
numFiles = len(fileNames)
annStart = 0

for k in range(0,numFiles):
    fileNameTemp = fileNames[k]
    # Find the image id from the filename
    fileNameTemp = fileNameTemp.rstrip('.jpg')
    fileNameTemp = fileNameTemp[len(anno_dir):]
    img_id = int(fileNameTemp)
    
    # Find all paths
    # Read in annotations for ICDAR 2011
    
    anno_file = open(anno_dir + "gt_" + repr(img_id) + ".txt")
   
    anno_list = anno_file.readlines()
    anno_img = []
    for i in range(0,len(anno_list)):
        anno_temp = anno_list[i]
        xmin,  ymin, xmax, ymax, word = anno_temp.split(",")
        word = word[1:len(word)-2]
        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)
        width = xmax-xmin
        height = ymax-ymin
        area = width*height
        bbox_use = [xmin, ymin, width, height]
        anno_img.append({"area":area,"bbox":bbox_use,"category_id":1,"id": annStart+i,"img_id":img_id,"utf8_string":word.lower()})
        
        test = 1
        
    img_anno = dict()
    img_file_name = repr(img_id) + '.jpg'
    
    with Image.open(anno_dir + img_file_name) as im:
        im_width, im_height = im.size
    im.close()
    
    img_anno["image"] = {"file_name": img_file_name,"height": im_height,"width": im_width,"id":img_id}
    img_anno["annotation"] = anno_img
            
    name = icdar_dataset + "_" + repr(img_id)  
    anno_file = "{}/{}.json".format(out_dir, name)
    with open(anno_file, "w") as f:
        # json.dump(img_anno, f, sort_keys=True, indent=2, ensure_ascii=False)
        s = json.dumps(img_anno, sort_keys=True, indent=2, ensure_ascii=False).encode('utf8')
        f.write(s)
        
    f.close()
    #annStart = annStart + len(anno_list)
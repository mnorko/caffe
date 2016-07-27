import argparse
from collections import OrderedDict
import json
import os
import sys
sys.path.append('..')

from coco_text import COCO_Text


parser = argparse.ArgumentParser(description = "Get the image size from an annotation file.")
parser.add_argument("annofile",
        help = "The file which contains all the annotations for a dataset in json format.")
parser.add_argument("imgsetfile", default = "",
        help = "A file which contains the image set information.")
parser.add_argument("namesizefile", default = "",
        help = "A file which stores the name size information.")


annofile = '/Users/marissac/data/coco/annotations/COCO_Text.json'
out_dir_train = '/Users/marissac/data/coco/annotations/combo_train2014'
out_dir_val = '/Users/marissac/data/coco/annotations/combo_val2014'
imgsetfile_train = '/Users/marissac/data/coco/ImageSets/combo_train2014.txt'
imgsetfile_val = '/Users/marissac/data/coco/ImageSets/combo_val2014.txt'
namesizefile_train = '/Users/marissac/caffe/data/coco/combo_train_name_size.txt'
namesizefile_val = '/Users/marissac/caffe/data/coco/combo_val_name_size.txt'


# initialize COCO api.
coco = COCO_Text(annofile)


if not os.path.exists(imgsetfile_train):
    print "{} does not exist".format(imgsetfile_train)
    sys.exit()
if not os.path.exists(imgsetfile_val):
    print "{} does not exist".format(imgsetfile_val)
    sys.exit()

name_size_dir = os.path.dirname(namesizefile_train)
if not os.path.exists(name_size_dir):
    os.makedirs(name_size_dir)
name_size_dir = os.path.dirname(namesizefile_val)
if not os.path.exists(name_size_dir):
    os.makedirs(name_size_dir)
    
# Read image info.
imgs_train = dict()
imgs_val = dict()
img_ids = coco.getImgIds()
for img_id in img_ids:
    # get image info
    img = coco.loadImgs(img_id)[0]
    
    file_name = img["file_name"]
    setLabel = img["set"]
    name = os.path.splitext(file_name)[0]
#    name_split = nameTemp.split("_")
#    name = "COCOText_" + setLabel + '_' + name_split[2]
    
    if setLabel == "train":
        imgs_train[name] = img
    else:
        imgs_val[name] = img

# Save name size information.
with open(namesizefile_train, "w") as nf:
    with open(imgsetfile_train, "r") as sf:
        for line in sf.readlines():
            name = line.strip("\n")
            img = imgs_train[name]
            nf.write("{} {} {}\n".format(img["id"], img["height"], img["width"]))
            
with open(namesizefile_val, "w") as nf:
    with open(imgsetfile_val, "r") as sf:
        for line in sf.readlines():
            name = line.strip("\n")
            img = imgs_val[name]
            nf.write("{} {} {}\n".format(img["id"], img["height"], img["width"]))

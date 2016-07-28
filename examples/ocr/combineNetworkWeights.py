# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 13:48:13 2016

@author: marissac
"""

import matplotlib.pyplot as plt
import sys
sys.path.append("/Users/marissac/caffe/examples/pycaffe/layers")

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Make sure that caffe is on the python path:
caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
import os
os.chdir(caffe_root)

import caffe

model_weights = '/Users/marissac/caffe/examples/ssd/models/VGGNet/cocoText/SSD_300x300/VGG_cocoText_legibleSplit_SSD_300x300_iter_244000.caffemodel'

synth_model = '/Users/marissac/caffe/examples/ocr/90ksynth/deploy_2.prototxt'
synth_model_weights = '/Users/marissac/caffe/examples/ocr/90ksynth/90ksynth_v2_v00_iter_140000.caffemodel'

model_combo = '/Users/marissac/caffe/examples/ocr/train_detectReadCombo.prototxt'

net_synth = caffe.Net(synth_model,synth_model_weights,caffe.TEST)
                     
net_combo = caffe.Net(model_combo,model_weights,caffe.TEST)

reader_list = [k for k, v in net_synth.params.items()]
combo_list = [k for k, v in net_combo.params.items()]

reader_params = {pr: (net_synth.params[pr][0].data, net_synth.params[pr][1].data) for pr in reader_list}
 
reader_keys = reader_params.keys()
for k in range(0,len(reader_keys)):
    for i in range(0,len(net_synth.params[reader_keys[k]])):
        net_combo.params[reader_keys[k]][i].data[...] = net_synth.params[reader_keys[k]][i].data

net_combo.save('/Users/marissac/caffe/examples/ocr/detect_2444000_read_140000_combo_temp.caffemodel')
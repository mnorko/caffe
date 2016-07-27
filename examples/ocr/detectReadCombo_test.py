# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 13:48:13 2016

@author: marissac
"""

import numpy as np
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
import multibox_util

from google.protobuf import text_format
from caffe.proto import caffe_pb2

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

#model_def = '/Users/marissac/caffe/examples/ocr/test_detectReadCombo_short.prototxt'
model_weights = '/Users/marissac/caffe/examples/ssd/models/VGGNet/cocoText/SSD_300x300/VGG_cocoText_legibleSplit_SSD_300x300_iter_244000.caffemodel'

synth_model = '/Users/marissac/caffe/examples/ocr/90ksynth/deploy_2.prototxt'
synth_model_weights = '/Users/marissac/caffe/examples/ocr/90ksynth/90ksynth_v2_v00_iter_140000.caffemodel'

combo_weights = '/Users/marissac/caffe/examples/ocr/detect_2444000_read_140000_combo_final.caffemodel'
model_combo = '/Users/marissac/caffe/examples/ocr/train_detectReadCombo.prototxt'
solver_combo = '/Users/marissac/caffe/examples/ocr/solver_detectReadCombo.prototxt'

#net = caffe.Net(model_def,model_weights,caffe.TEST)

#net_synth = caffe.Net(synth_model,synth_model_weights,caffe.TEST)
                     
net_combo = caffe.Net(model_combo,combo_weights,caffe.TEST)
#solver = caffe.SGDSolver(solver_combo)
#solver.net.copy_from(combo_weights)
                     
###detect_list = [k for k, v in net.params.items()]
#reader_list = [k for k, v in net_synth.params.items()]
#combo_list = [k for k, v in net_combo.params.items()]
##
##
###detect_params = {pr: (net.params[pr][0].data, net.params[pr][1].data) for pr in detect_list}
## 
#reader_params = {pr: (net_synth.params[pr][0].data, net_synth.params[pr][1].data) for pr in reader_list}
### 
##combo_params = {pr: (net_combo.params[pr][0].data, net_combo.params[pr][1].data) for pr in combo_list}   
###combo_params = {pr: (net_combo.params[pr][0].data) for pr in combo_list}   
#combo_params = {}   
#for i in range(0,len(combo_list)):
#    combo_params[combo_list[i]] = []
#    combo_params[combo_list[i]].append(net_combo.params[combo_list[i]][0].data)
#    if len(net_combo.params[combo_list[i]]) == 2:
#        #combo_params[combo_list[i]][0] = net_combo.params[combo_list[i]][0].data
#        combo_params[combo_list[i]].append(net_combo.params[combo_list[i]][1].data)
##  
#reader_keys = reader_params.keys()
#for k in range(0,len(reader_keys)):
#    #combo_params[reader_keys[k]][0] = reader_params[reader_keys[k]][0]
#    for i in range(0,len(net_synth.params[reader_keys[k]])):
#        net_combo.params[reader_keys[k]][i].data[...] = net_synth.params[reader_keys[k]][i].data
#        #net_combo.params[reader_keys[k]][1].data[...] = reader_params[reader_keys[k]][1]
#        if len(combo_params[reader_keys[k]])>i:
#            combo_params[reader_keys[k]][i] = net_synth.params[reader_keys[k]][i].data
#        else:
#            combo_params[reader_keys[k]].append(net_synth.params[reader_keys[k]][i].data)
##    if len(combo_params[reader_keys[k]]) == 2:
##        combo_params[reader_keys[k]][1] = reader_params[reader_keys[k]][1]
##    else:
##        combo_params[reader_keys[k]].append(reader_params[reader_keys[k]][1])
#
###combo_params['conv11'].append(reader_params['conv11'][1])
###combo_params2 = {pr: (net_combo.params[pr][0].data, net_combo.params[pr][1].data) for pr in combo_list}   
#net_combo.save('/Users/marissac/caffe/examples/ocr/detect_2444000_read_140000_combo_final.caffemodel')



net_combo.forward()    
#net_combo.backward()   
#solver.step(1)              
#net_synth.blobs['data'].data[...] = net_combo.blobs['data_transform'].data
#
#net_synth.forward()

#testMax = np.zeros((len(reader_keys),2))
#for k in range(0,len(reader_keys)):
#    for i in range(0,1):
#        net_vals = net_combo.params[reader_keys[k]][i].data
#        net_vals_flat = net_vals.reshape(-1)
#        net_reader_vals = net_synth.params[reader_keys[k]][i].data
#        net_reader_flat = net_reader_vals.reshape(-1)
#        net_diff = net_vals_flat-net_reader_flat
#        testMax[k,i] = max(net_diff)
#        if max(abs(net_diff)) != 0.0:
#            print reader_keys[k]
#            
#testMaxData = np.zeros((len(reader_keys)))
#for k in range(0,len(reader_keys)):
#    net_vals = net_combo.blobs[reader_keys[k]].data
#    net_vals_flat = net_vals.reshape(-1)
#    net_reader_vals = net_synth.blobs[reader_keys[k]].data
#    net_reader_flat = net_reader_vals.reshape(-1)
#    net_diff = net_vals_flat-net_reader_flat
#    testMaxData[k] = max(net_diff)
#    if max(abs(net_diff)) != 0.0:
#        print reader_keys[k]

output = net_combo.blobs['reshape'].data
words_final = []
for k in range(0,output.shape[0]):
    text_out = np.reshape(output[k,:,:,:],(39,23))
    text_max = np.argmax(text_out, axis=0) 
    output_word = ''
    for j in range(0,23):
        output_word = output_word + CAFFE_LABEL_TO_CHAR_MAP[text_max[j]-1]
        
    words_final.append(output_word)
####    img_transform = net.blobs['data_transform'].data[k,0,:,:]
####    plt.figure(k)
####    plt.imshow(img_transform)
    
data_out = net_combo.blobs['data_transform'].data 
for k in range(45,60):
    plt.figure(k)
    plt.imshow(data_out[k,0,:,:])
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 09:05:25 2016

@author: marissac
"""

import matplotlib
matplotlib.use('Agg')
 
import matplotlib.pyplot as plt
 
import pickle
import numpy as np
import re

import os


 
from argparse import ArgumentParser
parser = ArgumentParser('Plot test accuracy')
parser.add_argument('-o', '--output_plot_name', required=True,
                     help='path where output plot will be saved')
parser.add_argument('-l', '--train_log', required=False,
                     help='Path to train log file, from caffe on spark')
parser.add_argument('-f', '--foldername', type=str, required=True,
                    help='score log path')
# parser.add_argument('-t', '--plot_title', required=False,
#                     help='Title of plot')
 
def running_mean(x, N):
    return np.convolve(x, np.ones((N,)) / N, 'valid')
 
def parse_train_loss(trainlogfile):
    # Pick out lines of interest
    iteration = -1
    train_loss = []
    train_iters = []
    test_iters = []
    test_det = []
    regex_iteration = re.compile('Iteration (\d+)')
    regex_detection = re.compile('detection_eval')
    max_iter = 0
 
    with open(trainlogfile, 'r') as fh:
 
        fh.seek(0, 0)
        for line in fh.readlines():
            try:
                iteration_match = regex_iteration.search(line)
                det_match = regex_detection.search(line)
                if iteration_match:
                    if 'loss' not in line:
                        continue
                    iteration = float(iteration_match.group(1))
                    loss = float(line.strip().split('=')[1])
                    train_loss.append(loss)
                    train_iters.append(iteration / 1000.0)
 
                if det_match:
                    det_score = float(line.split(' ')[-1])
                    test_det.append(det_score)
                    test_iters.append(iteration / 1000.0)
 
                if max_iter < iteration / 1000.0:
                    max_iter = iteration / 1000.0
 
            except ValueError:
                # print line
                continue
    fh.close()
    return train_loss, train_iters, test_det, test_iters, max_iter
 
 
def parse_score_logs(foldername):
 
    detection_eval = []
    test_loss = []
    iters = []
    for file in os.listdir(foldername):
        iter = int(file.strip().split('_')[1])
        filepath = foldername + '/' + file
        fp = open(filepath, 'r')
 
        for line in fp:
            if 'detection_eval =' in line.strip():
                det = line.strip().split('=')[1]
                detection_eval.append(float(det))  # * 100)
                iters.append(float(int(iter) / 1000.0))
            if 'loss =' in line.strip():
                loss = line.strip().split('=')[1]
                test_loss.append(float(loss))
 
    return detection_eval, test_loss, iters
 
 
root_dir = "/Users/marissac/caffe"
output_plot_name = root_dir + "/examples/ssd/jobs/VGGNet/cocoText/SSD_300x300/results/cocoText_legibleSplit_testErr"
train_log = root_dir + "/examples/ssd/jobs/VGGNet/cocoText/SSD_300x300/VGG_cocoText_legibleSplit_SSD_300x300_full.log"
foldername = "/Users/marissac/Code/text_reading/ssd/jobs/cocoText/SSD_300x300/test_logs_legible"
  
train_loss, train_iters, test_det, test_iters_bad, max_iter = parse_train_loss(train_log)
detection_eval, test_loss, test_iters = parse_score_logs(foldername)
 
train_running_avg_loss = running_mean(train_loss, 100)
 
offset = len(train_loss) - len(train_running_avg_loss)
train_iters_running = train_iters[offset:]
print len(train_running_avg_loss), len(train_iters_running)
 
if len(test_loss) != len(detection_eval):
    del test_loss[-1]
print len(detection_eval), len(test_loss)
 
 
num_iter = len(train_iters)
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(train_iters, train_loss, '.', color='skyblue', label='training loss')
ax2.plot(test_iters, detection_eval, 'ro', label='Detection eval mAP')
ax1.plot(test_iters, test_loss, 'go', label='Test Loss')
ax1.plot(train_iters_running, train_running_avg_loss, 'b.', label='train running avg loss')
 
ax1.set_xlabel('Iteration * 10^3')
ax1.set_ylabel('Training Loss', color='b')
ax2.set_ylabel('Test mAP', color='r')
 
ax1_max = max(60.0, max_iter)
# ax1_max = 6.0
ax1.axis([0 , ax1_max , 0.0, 6.0])
ax2.axis([0, ax1_max, 0.0, 1.0])
ax2.set_yticks(np.arange(0.0, 1.0, 0.02), minor=True)
ax2.yaxis.set_ticks(np.arange(0.0, 1.0, 0.05))
ax2.grid(which='minor', alpha=0.3)
ax2.grid(which='major', alpha=0.7)
 
 
# plt.legend(handles=[ test_handle, train_handle, running_handle], prop={'size':12})
plt.legend(shadow=True, fancybox=True, prop={'size':12})
plotname = output_plot_name + '_loss.jpg'
plt.savefig(plotname)
print 'saved plot to : ', plotname
 

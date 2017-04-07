#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import os, glob
import argparse
caffe_root = '/home/nathan/caffe-segnet/'
import sys
import cv2
import shutil
sys.path.insert(0, caffe_root + 'python')

import caffe


# Test tile-wise classification

'''
pseudocode::

intialize network

load image data

preprocessing procedure

feed data into network

obtain results - including intermediates

'''

def preprocess(img):
	pass



def load_data(path):
	pass



def push_image(net, img):
	pass



def run(model, weights, source):
	caffe.set_mode_gpu()

	net = caffe.Net(model, weights, 'test')





if __name__=='__main__':

	model = ''
	weights = ''

	run(model, weights, source)







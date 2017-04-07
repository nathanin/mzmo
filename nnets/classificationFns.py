#!/usr/bin/python

# Imports
import sys
import os
import cv2
import glob
import numpy as np
import itertools
import time

import matplotlib.pyplot as plt
CAFFE_ROOT = '/Users/nathaning/caffe-segnet-cudnn5'
#CAFFE_ROOT = '/home/nathan/caffe-segnet-cudnn5'
sys.path.insert(0, CAFFE_ROOT+'/python')

import caffe


# Function definition
def define_network(model, weights):
	caffe.set_mode_gpu() # CPU mode for my feeble laptop.
	net = caffe.Net(model, weights, caffe.TEST)
	
	return net

def get_img(path, s=256):
	'''
	Return an array ready for caffe to chew.
	
	TODO handle resizing by pulling params from a caffe.Net object
	'''
	img = cv2.imread(path)
	img = cv2.resize(img, dsize = (s, s))
	img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
	img = np.swapaxes(img, 0, 2)
	img = np.asarray([img])
	return img

def infer(net, data, outputlayer = "prob"):
	'''
	Process the data contained in data;
	
	TODO handle batch-size > 1 : check data matches expected data in net
	'''
	out = net.forward_all(data = data)
	label = out[outputlayer][0].argmax(axis=0)

	return net, label

def infer_load_img(net, path, s):
	'''
	Combination of infer() and get_img()
	
	'''
	img = get_img(path, s)
	net, label = infer(net, img)
	
	return label

def infer_return_layer(net, path, s, layer_name):
	'''
	1. Load image
	2. Run net forward
	3. Return a blob identified by layer_name
	'''
	
	img = get_img(path, s)
	net, label = infer(net, img)
	blob = net.blobs[layer_name].data
	return np.squeeze(blob) # Not sure if it should ALWAYS squeeze but for now its OK.
	

def list_images(path, exts = ['*.tif','*.jpg']):
	'''
	Return iterable object with the full paths of image files in path
	
	TODO how to return a normal list from itertools.chain() object
	'''
	files = []
	for p in exts:
		match = os.path.join(path, p)
		files.append(glob.glob(match))

	return list(itertools.chain.from_iterable(files))

def get_blob_slice(net, layer_name, level):
	'''
	Blobs are 4-D tensors:
	BxKxHxW
	- B is batch size (assume = 1)
	- K is 3rd dimension of image; equal to the number of filters for Convolution layers (??)
	- H and W are image H and W. They are -- apparently -- permuted w.r.t cv2.imread() orientation.

	^^^ check that one. 
	'''
	blob = net.blobs[layer_name].data
	blob_slice = blob[:,level,:,:]
	blob_slice = np.squeeze(blob_slice)
	blob_slice = np.swapaxes(blob_slice, 0, 1)

	return blob_slice


def get_blob_argmax(net, layer_name):
	'''
	Same as above, but instead of slicing into the blob, compute the argmax

	'''
	blob = net.blobs[layer_name].data
	blob_max = np.argmax(blob, 1)
	blob_max = np.squeeze(blob_max)
	blob_max = np.swapaxes(blob_max, 0, 1)

	return blob_max


def get_blob_amax(net, layer_name):
	'''
	Same as above, but instead of slicing into the blob, compute the argmax

	'''
	blob = net.blobs[layer_name].data
	#blob_max = np.argmax(blob, 1)
	blob_max = np.amax(blob, 1)
	blob_max = np.squeeze(blob_max)
	blob_max = np.swapaxes(blob_max, 0, 1)

	return blob_max

def plot_slices(net, layer_name, slice_range = None, interval = 10):
	'''
	Plot all slices of a layer, layer_name, as if a movie. 
	'''
	fig = plt.figure()
	a = fig.add_subplot(1,1,1)
	blob = net.blobs[layer_name].data
	
	b,k,h,w = blob.shape
	if slice_range == None:
		slice_range = range(k)

	for layer in slice_range:
		slice_data = get_blob_slice(net, layer_name, layer)
		SLICE_PLOT = plt.imshow(slice_data)
		a.set_title('{} layer {}'.format(layer_name, layer))
		#time.sleep(interval)
		plt.pause(0.5)

def get_img_from_net(net):
	img = net.blobs['data'].data #(1,3,227,227)
	img = np.squeeze(img)
	img = np.swapaxes(img, 0, 2)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img = cv2.convertScaleAbs(img)
	return img


def cast_img_to_net_ready(img, s):
	''' TODO: take in a network and infer the needed shape from 'data' '''
	# Take in as (H, W, d) make it (d, H, W)
	img = cv2.resize(img, dsize = (s, s))
	img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
	img = np.swapaxes(img, 0, 2)
	
	# Caffe takes care of conversion between uint and double float. 	
	return img 


def plot_data_layer(net):
	img = get_img_from_net(net)
	plt.imshow(img)
	#fig.colorbar(cax, ticks = get_range(data),  orientation='horizontal')
	#return data


def plot_data_segmentation(net):
	fig, axarr = plt.subplots(1,2)
	img = get_img_from_net(net)
	label = net.blobs['prob'].data
	label = np.squeeze(label[0,:,:,:])
	label = np.argmax(label, axis=0)
	label = np.swapaxes(label, 0,1)
	
	axarr[0].set_title('Data')
	axarr[1].set_title('Label')
	axarr[0].imshow(img)
	axarr[1].imshow(label, cmap = 'gray')

	#fig.colorbar(ticks= [0,25,50,75], ax = axarr.ravel().tolist())


def plot_conv_layer(net, layer):
	fig, axarr = plt.subplots(1,2)
	fn = ['amax', 'argmax']
	print '{}, {}'.format(layer, net.blobs[layer].data.shape)
	for ax, f in zip(axarr.flat, fn):
		if f == 'amax':
			data = get_blob_amax(net, layer)
			im = ax.imshow(data, cmap = 'gray')
		elif f == 'argmax':
			data = get_blob_argmax(net, layer)
			im = ax.imshow(data)

		ax.set_title('{} {}'.format(layer, f))

	#fig.colorbar(im, ax = axarr.ravel().tolist(), orientation = 'horizontal')


def plot_layer_pairs(net, layers):
	fig, axarr = plt.subplots(1,4)
	#plt.rcParams['figure.figsize'] = [9,29]
	fn = ['amax', 'argmax', 'amax', 'argmax']	
	for ax, l, f in zip(axarr.flat, sorted(layers+layers), fn):
		if f == 'amax':
			data = get_blob_amax(net, l)
			im = ax.imshow(data, vmax=1, vmin=0)
		elif f == 'argmax':
			data = get_blob_argmax(net, l)
			im = ax.imshow(data)

		ax.set_title('{} {}'.format(l, f))

	#fig.colorbar(im, ax = axarr.ravel().tolist(), orientation = 'horizontal')
		

def get_range(blob):
	'''
	Return a range of 4 numbers between the max and min in a blob
	'''	
	hi = np.amax(blob)
	lo = np.amin(blob)
	return np.arange(lo, hi, (hi-lo)/4)


def plot_layers(net, layer_names = None):
	'''
	Cleaned version of ^ ; makes more assumptions.
	'''
	if layer_names == None:
		error('Layer names undefined')

	plt.rcParams['figure.figsize'] = [10, 5]
	print 'Plotting {} figures'.format(len(layer_names))	
	for layer in layer_names:
		if 'data_segmentation' in layer:
			#print 'data and segmentation layer'
			plot_data_segmentation(net)
		if layer == 'data':
			#print 'data layer only'
			plot_data_layer(net)
		if 'conv' in layer:
			#print 'convolution layer only'
			plot_conv_layer(net, layer)	
		if len(layer) == 2:
			#print 'layer pair'
			plot_layer_pairs(net, layer)
	plt.show()


# def vis_square(data):
# 	'''
# 	From caffe-master example ipython notebook: //start
# 	Take an array of shape (n, height, width) or (n, height, width, 3) and visualize each
# 	(height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
# 	'''
# 	# normalize data for display
# 	data = (data - data.min()) / (data.max() - data.min()) # Median centering
	
# 	# force the number of filters to be square
# 	n = int(np.ceil(np.sqrt(data.shape[0])))
# 	padding = (((0, n ** 2 - data.shape[0]), 
# 		   (0,1), (0,1))	# add some space between filters 
# 		   + ((0,0),) * (data.ndim - 3)) # don't pad the last dimension (if there is one)
# 	data = np.pad(data, padding, mode = 'constant', constant_values = 1)

# 	# tile the filters into an image
# 	data = data.reshape((n,n) + data.shape[1:].transpose((0,2,1,3) + tuple(range(4, data.ndim + 1))))
# 	data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

# 	plt.imshow(data)

def vis_square(data, bwmode = False):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""
    
    # normalize data for display -- median center at 0. 
    data = (data - data.min()) / (data.max() - data.min())
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
               (0, 1), (0, 1))                 # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    if bwmode:
	plt.imshow(data, cmap='gray'); plt.axis('off')
    else: 
	print data.shape
        plt.imshow(data); plt.axis('off')




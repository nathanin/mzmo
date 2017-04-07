#!/usr/bin/python

'''

Based on the Camvid example,
Code for running the trained .caffemodel file on a set of images
arguments given at command line. 

/home/nathan/semantic-pca/code/test_tiles_pullimgs.py

'''


# Clean up using my new knowledge

import numpy as np
import matplotlib.pyplot as plt
import os, glob
import argparse
# from sklearn.preprocessing import normalize
caffe_root = '/home/nathan/caffe-segnet/'
#caffe_root = '/home/ubuntu/caffe-segnet/'
import sys
import cv2
import shutil
sys.path.insert(0, caffe_root + 'python')

import caffe

def define_colors():
	c1 = [45, 	55, 220] # 0 
	c2 = [45, 	55, 220] # 1 
	c3 = [45, 	55, 220] # 2 ^ all the same color
	c4 = [255, 255, 255] # 3 St

	label_colours = np.array([c1, c2, c3, c4])
	label_names = ("G3", "G4", "BN", "St")

	return label_colours, label_names

def color_image(rgb, path, name):
	img = cv2.imread(os.path.join(path, name))

	img = np.add(img*0.5, rgb*0.5)
	img = cv2.convertScaleAbs(img)

	return img

def run(model, weights, tiledir, write_rgb, 
		write_labels = None, write_prob = None, originaldir = '', cleanup = False):
	write_labels_bool = os.path.exists(write_labels)
	write_prob_bool = os.path.exists(write_prob)

	caffe.set_mode_gpu()

	if cleanup:
		"Cleaning destinations"
		shutil.rmtree(write_prob)
		shutil.rmtree(write_labels)
		shutil.rmtree(write_rgb)

		os.makedirs(write_prob)
		os.makedirs(write_labels)
		os.makedirs(write_rgb)

	net = caffe.Net(model,
	                weights,
	                caffe.TEST)

	# Color definitions
	[label_colours, label_names] = define_colors()

	# search_term = os.path.join(tiledir, '*.jpg')
	# img_list = glob.glob(search_term) # UNSORTED, SORTED, OR PULL IT FROM list.txt <<-- that one
			
	list_file = os.path.join(tiledir, 'list.txt')
	print list_file
	img_list = []
	with open(list_file, 'r') as f:
		for ln in f:
			ln = ln.split()
			img_list.append(ln[0])

	nims = len(img_list)
	classes = np.zeros((nims, len(label_colours)))
	print classes.shape

	# Same length and order as the list.txt
	for n, img in enumerate(img_list):
		# print os.path.basename(img), 
		img_name = os.path.basename(img)
		img_data = cv2.imread(img)

		out = net.forward()

		predicted = net.blobs['prob'].data
		output = np.squeeze(predicted[0,:,:,:]) # first one
		ind = np.argmax(output, axis=0) # `ind` is the label image. `output` is a matrix of probs. 
		
		r = ind.copy()
		g = ind.copy()
		b = ind.copy()

		# Impose the colors on our copied label matrices.
		for l in range(len(label_colours)):
			x = (ind==l).sum()
			classes[n, l] = x

			r[ind==l] = label_colours[l,0]
			g[ind==l] = label_colours[l,1]
			b[ind==l] = label_colours[l,2]

		rgb = np.zeros(shape=(ind.shape[0], ind.shape[1], 3), dtype=np.uint8)
		rgb[:,:,2] = r#/255.0
		rgb[:,:,1] = g#/255.0
		rgb[:,:,0] = b#/255.0


		IMAGE_WRITE = img_name.replace('.jpg', '.png')
		IMAGE_WRITE = os.path.join(write_rgb, IMAGE_WRITE)
		
		if os.path.exists(tiledir):
			rgb = color_image(rgb, tiledir, img_name)

		cv2.imwrite(filename = IMAGE_WRITE, img = rgb)


		if write_labels_bool:
			LABEL_WRITE = img_name.replace('.jpg', '.png')
			LABEL_WRITE = os.path.join(write_labels, LABEL_WRITE)
			cv2.imwrite(filename = LABEL_WRITE, img = ind)

		if write_prob_bool:
			# Output all classes now. img_p1.jpg, img_p2.jpg, etc.
			for c, _ in enumerate(label_names):
				PROB_WRITE = img_name.replace('.jpg', '_p{}.png'.format(c))
				PROB_WRITE = os.path.join(write_prob, PROB_WRITE)
				prob = output[c,:,:]
				cv2.imwrite(filename = PROB_WRITE, img = prob*255) # Scale


 		if n % 200 == 0:
 			print os.path.basename(img), 
 			img_area = nims*256*256
			for l in range(len(label_names)):
					x = classes[:, l]
					print "{}:{}\t".format(label_names[l], str(x.sum()/img_area)),
			print "Writing tile {:06d} / {:06d}".format(n, nims)


	print "========== totals =========="
	img_area = nims*256*256
	for l in range(len(label_names)):
		x = classes[:, l]
		print "{}:{}\t".format(label_names[l], str(x.sum()/img_area)),
	print '\nDone'

if __name__ == '__main__':
	# Import arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', type=str, required=True)
	parser.add_argument('--weights', type=str, required=True)
	parser.add_argument('--tiledir', type=str, required=True)
	parser.add_argument('--write_rgb', type=str, required=True)
	parser.add_argument('--originaldir', type=str, required=False)
	parser.add_argument('--write_prob', type=str, required=False)
	parser.add_argument('--write_labels', type=str, required=False)

	args = parser.parse_args()

	model = os.path.expanduser(args.model)
	weights = os.path.expanduser(args.weights)
	tiledir = os,path.expanduer(args.tiledir)

	if args.write_labels:
		write_labels = args.write_labels
	else:
		write_labels = ''

	if args.write_prob:
		write_prob = args.write_prob
	else:
		write_prob = ''

	if args.logfile:
		logfile = args.logfile
	else:
		logfile = None

	if args.originaldir:
		originaldir = args.originaldir
	else:
		originaldir = ''

	run(model 		= model, 
		weights 	= weights, 
		tiledir 	= tiledir, 
		write_rgb 	= write_rgb,
		write_labels= write_labels,
		write_prob 	= write_prob,
		originaldir = originaldir)








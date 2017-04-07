#!/usr/bin/python

# With logging.
# /home/nathan/mzmo/code/pretrained/collect_data.py
import sys
import os, glob
import cPickle as pickle
import pandas as pd
import re
import numpy as np

# sys.path.insert(0, '/home/nathan/mzmo/code')
import classificationFns as cf
sys.path.insert(0, '/home/nathan/caffe-segnet-cudnn5')
import caffe
print 'imports successful'

# #################################################################
# #################################################################
# ####
# #### 	Gather data
# ####
# #################################################################
# #################################################################

# def get_casenames(labels):
# 	casenames = [re.split('^SP ', c)[1] for c in labels]
# 	casenames = [re.split('\s', c, 2)[0] for c in casenames]
# 	casenames = np.asarray(casenames)
# 	ucases = np.unique(casenames)

# 	return casenames, ucases

# def get_grades(labels):
# 	grades = [re.split('\(', s)[1] for s in labels]
# 	grades = [re.split('\)', s)[0] for s in grades]
# 	grades = [re.sub('[^\d]*$', '', s) for s in grades]
# 	# for g in grades:
# 	# 	print g
# 	grades = np.asarray(grades)
# 	ugrades = np.unique(grades)

# 	return grades, ugrades


def gather_data(fnames, net, outname, layer, maxentries=30000):
    # dropbox = '/home/nathan/Dropbox/projects/mzmo/data/SFTA_GABOR'
    # fnames = ['M0RRP_1.csv', 'M0XRT_1.csv', 'M0RRP_2.csv']
    x = []
    labels = []

    print 'Searching {}'.format(fnames)
    labels = glob.glob(os.path.join(fnames, '*.jpg'))
    labels = [os.path.basename(f) for f in labels]

    print 'Found {} files'.format(len(labels))
    if len(labels) > maxentries:
        # print 'Found {} files'.format(len(labels))
        labels = np.random.choice(labels, maxentries)
        # labels = labels[idx]
    print 'Found {} files'.format(len(labels))

    for f in labels:
        print os.path.join(fnames, f)
        img = cf.get_img(os.path.join(fnames, f), 227)
        net.forward_all(data=img)
        feat = net.blobs[layer].data
        # print feat[:20]
        x.append(feat.flatten())

    x = np.vstack(x)
    # labels = list(np.concatenate(labels))
    print 'Concatenated {} training files into: {}'.format(len(labels), x.shape)

    print 'Saving {}'.format(outname)
    f = open(outname, 'w')
    pickle.dump(x, f)
    pickle.dump(labels, f)
    # cPickle.dump(casemap, f)
    # cPickle.dump(grademap, f)
    f.close()


# print 'saving numpy'
# np.save('m0_test.npy',data)

#################################################################
#################################################################
if __name__ == '__main__':

    project = '/home/nathan/mzmo'
    # project = '/Users/nathaning/_projects/mzmo'
    net_proto = os.path.join(project, 'bvlc_alexnet', 'deploy.prototxt')
    print 'Network prototxt: {}'.format(net_proto)

    net_weights = os.path.join(project, 'bvlc_alexnet',
                               'bvlc_alexnet.caffemodel')
    print 'Network weights: {}'.format(net_weights)

    net = cf.define_network(net_proto, net_weights)
    print 'Net defined successfully.'

    layer = 'fc7'

    # m0_dir = '/home/nathan/mzmo/data/source_feature/256/0'
    # m1_dir = '/home/nathan/mzmo/data/source_feature/256/1'
    m0_dir = '/home/nathan/mzmo/data/nuclei/indiv.interior.2/0'
    m1_dir = '/home/nathan/mzmo/data/nuclei/indiv.interior.2/1'
    m0_write = '/home/nathan/mzmo/data/nuclei/indiv.interior.2/m0_fc7.p'
    m1_write = '/home/nathan/mzmo/data/nuclei/indiv.interior.2/m1_fc7.p'

    gather_data(m0_dir, net, m0_write, layer)
    print '#################################################################\n' * 3
    print ''

    gather_data(m1_dir, net, m1_write, layer)
    print '#################################################################\n' * 3
    print ''

#################################################################
#################################################################

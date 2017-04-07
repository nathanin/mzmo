#!/home/ingn/miniconda2/bin/python

# With logging.

import sys
import os
import cPickle
import pandas as pd
import h5py
import re

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import (LassoCV, ElasticNetCV)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import (train_test_split, cross_val_score, KFold,
                                     StratifiedKFold)

from sklearn.feature_selection import (VarianceThreshold, SelectKBest,
                                       mutual_info_classif, f_classif,
                                       SelectFromModel, SelectPercentile)

from sklearn import decomposition
from sklearn.manifold import (Isomap, TSNE)

# #################################################################
# #################################################################
# ####
# #### 	Gather data
# ####
# #################################################################
# #################################################################


def get_casenames(labels):
    casenames = [re.split(r'^SP[ _]', c)[1] for c in labels]
    casenames = [re.split(r'[\s_]', c, 2)[0] for c in casenames]
    casenames = np.asarray(casenames)
    ucases = np.unique(casenames)

    return casenames, ucases


def get_grades(labels):
    grades = [re.split(r'\(', s)[1] for s in labels]
    grades = [re.split(r'\)', s)[0] for s in grades]
    grades = [re.findall(r'^(\D|[3-5]){1}', s)[0] for s in grades]
    # for g in grades:
    # 	print g
    grades = np.asarray(grades)
    ugrades = np.unique(grades)

    return grades, ugrades


#TODO fix hard-coded source dir (Nathan)
def gather_data(fnames, outname):
    source_dir = '/home/ingn/mzmo/data/nuclei/indiv.fixed64.pick20.nodilate/AG/63f'
    # fnames = ['M0RRP_1.csv', 'M0XRT_1.csv', 'M0RRP_2.csv']
    x = []
    labels = []
    cases = []
    for f in fnames:
        f = os.path.join(source_dir, f)
        #data = pd.read_csv(f, delimiter = ',', header = None)
        data = pd.read_excel(f)
        data = data.fillna(method='ffill')
        print 'Loaded data: {} {}'.format(f, data.shape)

        x.append(data.values[:, 2:])  # usually 1:
        labels.append(data.values[:, 0])

    x = np.concatenate(x)

    labels = list(np.concatenate(labels))
    print 'Concatenated {} M0 training files into: {}'.format(
        len(fnames), x.shape)

    print 'Mapping cases together'
    patt = r'\d\s[1-3A-Z][1-3A-Z].+$'
    casenames, ucases = get_casenames(labels)
    grades, ugrades = get_grades(labels)

    casemap = {}
    for i, c in enumerate(ucases):
        indices = np.where(casenames == c)
        casemap[c] = indices[0]

    grademap = {}
    for i, g in enumerate(ugrades):
        indices = np.where(grades == g)
        grademap[g] = indices[0]

    tot = 0
    for i, c in enumerate(casemap.iterkeys()):
        ix = casemap[c]
        tot += ix.shape[0]
        print '{:3d} Case ID: {:20s} ~ Index Range: {:4d}-{:4d} {} total = {}'.format(
            i, c, ix[0], ix[-1], ix.shape, tot)

    print 'GRADES: '
    tot = 0
    for i, c in enumerate(grademap.iterkeys()):
        ix = grademap[c]
        tot += ix.shape[0]
        print '{:3d} String: {:20s} ~ Number: {} \ttotal = {}'.format(
            i, c, ix.shape, tot)

    # these are all "0"
    # y = np.zeros(shape=(x.shape[0], 1))

    # data = np.concatenate((x,y), axis = 1)

    outname = os.path.join(source_dir, outname)
    print 'Saving {}'.format(outname)
    f = open(outname, 'w')
    cPickle.dump(x, f)
    cPickle.dump(labels, f)
    # cPickle.dump(casemap, f)
    # cPickle.dump(grademap, f)
    f.close()


# print 'saving numpy'
# np.save('m0_test.npy',data)

if __name__ == '__main__':
    #################################################################
    #################################################################
    m0_name = sys.argv[1]
    m0_out = sys.argv[2]
    m1_name = sys.argv[3]
    m1_out = sys.argv[4]

    print 'm0_name: {}'.format(m0_name)
    print 'm0_out: {}'.format(m0_out)
    print 'm1_name: {}'.format(m1_name)
    print 'm1_out: {}'.format(m1_out)

    fnames = [m0_name]
    gather_data(fnames, m0_out)
    print '#################################################################\n' * 2
    fnames = [m1_name]
    gather_data(fnames, m1_out)
    print '#################################################################\n' * 2

    #################################################################
    #################################################################

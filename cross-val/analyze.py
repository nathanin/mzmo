#!/home/ingn/miniconda2/bin/python

# project and /or analyze features in the sfta/gabor folder

import numpy as np
import os, sys
import pickle
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

# Projection functions
from sklearn import decomposition
from sklearn.manifold import (Isomap, TSNE)  # import defnet

print 'package importing successful'

####
## /home/ingn/mzmo/pretrained/analyze.py
####

# #################################################################
# #################################################################
# ####
# #### 	Define some classes
# ####
# #################################################################
# #################################################################


class Tile:
    def __init__(self, data, case, grade, label, m):
        self.data = data
        self.case = case
        self.grade = grade
        self.label = label
        self.m = m

    def print_info(self):
        print 'm{}\t{:7s}\t{:9s}\t{:2s}\t\"{:45s}\"'.format(
            self.m, self.data.shape, self.case, self.grade, self.label)

    def infostr(self):
        return 'm{}  {:7s}   {:9s}   {:2s}   \"{:45s}\"'.format(
            self.m, self.data.shape, self.case, self.grade, self.label)


# #################################################################
# #################################################################
# ####
# #### 	Dataset class
# ####
# #################################################################
# #################################################################


class Dataset:
    '''

	TODO store classification results as an attr

	'''
    # Immutable attribute
    grade_dict = {
        '3': 6,
        '3+3': 6,
        '3+4': 7,
        '3+4+5': 9,
        '3+5': 8,
        '3,4+5': 9,
        '4': 8,
        '4+3': 7,
        '4+4': 8,
        '4+5': 9,
        '5': 10,
        '5 5': 10,
        '5+3': 8,
        '5+4': 9,
        '5+5': 10,
        '5_5': 10
    }

    def __init__(self, data):
        # Data is a list of Tile()'s
        self.data = data

    def filter_out(self, param, rule):
        print 'Filtering out points where {} is {}'.format(param, rule)
        orig = len(self.data)
        self.data = [d for d in self.data if getattr(d, param) != rule]
        newlen = len(self.data)
        print 'Filtered out {} points\n'.format(orig - newlen)

    def filter(self, param, rule):
        self.data = [d for d in self.data if getattr(d, param) in rule]

    def list_param(self, param):
        u = [getattr(k, param) for k in self.data]
        u = np.asarray(u)  # not sure if needed
        u = np.unique(u)
        return list(u)

    def list_param_all(self, param):
        u = [getattr(k, param) for k in self.data]
        return u

    def enumerate_param(self, param):
        plist = self.list_param(param)
        # now make it a dict:
        d = {}
        for i, k in enumerate(plist):
            d[k] = i
        return d

    def append_gleason(self, indata):
        # Append custom data, for now just do the grade encoding
        gr = []
        for d in self.data:
            gr.append(self.grade_dict[d.grade])
        gr = np.asarray(gr)
        gr.shape = (gr.shape[0], 1)  # add non-zero dimension......

        indata = np.concatenate((indata, gr), axis=1)
        return indata

    def print_info(self):
        print 'Data: {}'.format(len(self.data))

    def print_class_pct(self, param):
        p = self.list_param_all(param)
        up = self.list_param(param)

        for p_ in up:
            print '{}: {}'.format(p_, (p == p_).sum())

    def hold_out(self, filterparam, skipvalue):
        d_train = []
        d_held = []
        for k in self.data:
            p = getattr(k, filterparam)
            if p == skipvalue:
                d_held.append(k)
            else:
                d_train.append(k)
        return Dataset(d_train), Dataset(d_held)

    def build(self, labelparam, ydict=None):
        '''
		Return data in the form:
			x: features from each data.data[] vector
			y: enumerated labels of the labelparam
			ydict: the dictionary according to which y is enumerated
                -- Useful for performing classification against
                   some arbitrary parameter of the data
		'''
        x = []
        y = []
        if ydict is None:
            ydict = self.enumerate_param(labelparam)
        for k in self.data:
            #x = np.vstack((x, k.data))
            x.append(k.data)
            y.append(ydict[getattr(k, labelparam)])
        #x = np.concatenate(x)
        #print 'Concatenating {} elements each of shape {}'.format(
        #    len(x), x[0].shape)
        x = np.vstack(x)
        y = np.asarray(y)
        return x, y, ydict

    def gather_by(self, param):
        # return a dict
        # d[x1] = [Datum1, Datum2, etc.]
        # d[x2] = ....
        # Precursor to averaging all tiles/case, for instance.
        d = {}
        u = [getattr(k, param) for k in self.data]
        u = np.asarray(u)
        u = np.unique(u)
        for k in u:
            d[k] = []
        for k in self.data:
            d[getattr(k, param)].append(k)
        return d

    def average_all(self, param):
        # Return a dataset
        # 1. Gather all tiles of same 'param' value
        # 2. For each unique 'param':
        #   - Create a new "Tile"
        #   - Inherit all aspects of the parent except for data
        #   - Set data to be the mean of all the members of 'param'
        gathered = self.gather_by(param)
        print "length gathered = {}".format(len(gathered))
        #print "length gathered[0] = {}".format(len(gathered[0]))
        averaged = []
        for k in gathered.iterkeys():
            #print 'key: {}'.format(k)
            p = gathered[k]
            data = [t.data for t in p]
            #print len(data)
            data = np.vstack(data)
            #print data.shape
            data = np.mean(data, axis=0)  # ????
            #print data.shape
            t = p[0]
            #print 'Adding point: {}'.format(t.infostr())
            data.shape = t.data.shape
            #print data.shape
            t.data = data
            averaged.append(t)  # >>> ??????????

        print 'Creating new dataset from {} tiles'.format(len(averaged))
        return Dataset(averaged)


# #################################################################
# #################################################################
# ####
# #### 	Gather data
# ####
# #################################################################
# #################################################################


# NEXT TODO (Nathan)
def get_case_tile(label):
    pass


def get_casename(label):
    casename = re.split('^SP[ _]', label)[1]
    # print '{}\t'.format(casename),
    casename = re.split('[\s_]', casename, 2)[0]
    # print '{}\t'.format(casename)

    # print casename
    return casename


def get_grade(label, infostring):
    # If one of the special ones, return it
    # otherwise parse it based on the format of `label`
    if 'SC' in label:
        grade = 'SC'
    elif 'NE' in label and 'SIG' not in label:
        grade = 'NE'
    elif 'N20' in infostring or 'nuclei' in infostring:
        grade = re.split(r'\(', label)[1]
        # print '{}\t'.format(grade),
        grade = re.split(r'\)', grade)[0]
        # print '{}\t'.format(grade),
        grade = re.sub(r'\D+$', '', grade)
        # print '{}\t'.format(grade)
    else:
        grade = re.split(r'_{2}', label)[1]
        grade = re.split(r'_{3}', grade)[0]
        grade = re.sub(r'[^\d]*$', '', grade)
        grade = re.sub(r'_', ' ', grade)
        # These two have an extra '_' somewhere
        if grade == '4 L3 005' or grade == '4 L3 003':
            grade = '5+5'

    #if len(grade) >= 1:
    #    return grade
    #else:
    #    return None
    return grade


'''
Return a list of Tile() objects
with data read from strings
'''


def populate_tiles(fpath, m, infostring):
    print 'Populate tiles from : {}'.format(fpath)
    f = open(fpath, 'r')
    contents = []
    x, l = [pickle.load(f) for _ in range(2)]
    for i, l_ in enumerate(l):
        # print l_
        cn = get_casename(l_)
        gr = get_grade(l_, infostring)
        data = x[i, :]
        data.shape = (1, data.shape[0])
        contents.append(Tile(data=data, case=cn, grade=gr, label=l_, m=m))
    return contents


# #################################################################
# #################################################################
# ####
# #### 	Balancing for uneven classes
# ####
# #################################################################
# #################################################################


def balance(x, y):
    # get x as classes in y
    x_ = {}
    yclasses = np.unique(y)
    print 'unique classes {}'.format(yclasses)
    m = []
    for cl in yclasses:
        x_[cl] = x[y == cl, :]
        print 'x[{}].shape {}'.format(cl, x[y == cl, :].shape)
        m.append((y == cl).sum())
    m = np.asarray(m)
    print 'count of each class {}'.format(m)
    small = np.argmin(m)
    large = np.argmax(m)
    print 'small: {}({}), large {}({})'.format(small, m[small], large, m[large])
    cvect = np.arange(0, m[large])
    cvect = np.random.choice(cvect, m[small])
    print 'x[{}] before {} ---> '.format(large, x_[large].shape),
    x_[large] = x_[large][cvect, :]
    print 'after {}'.format(x_[large].shape)
    xout = np.concatenate((x_[small], x_[large]))
    yout = np.concatenate((np.repeat(small, m[small]), np.repeat(
        large, m[small])))

    print 'xshape: {}'.format(xout.shape)
    print 'y=0: {}, y=1: {}'.format((yout == 0).sum(), (yout == 1).sum())
    return xout, yout


def balance_dataset(dset, param):
    #dset.print_class_pct(param)
    _, y, ydict = dset.build(param)
    # get class percents
    m = []
    for k in ydict.iterkeys():
        m.append((y == ydict[k]).sum())
    m = np.asarray(m)

    small = np.argmin(m)
    large = np.argmax(m)
    ratio = m[small] / float(m[large])

    # go through dset and for each case throw away tiles
    newset = []
    dsetcases = dset.gather_by('case')
    for dsc in dsetcases.iterkeys():
        tlist = dsetcases[dsc]
        if ydict[getattr(tlist[0], param)] == large:
            n = np.ceil(len(tlist) * ratio).astype(np.int)
            tlist = np.random.choice(tlist, n)
        for t in tlist:
            newset.append(t)

    return Dataset(newset)


# #################################################################
# #################################################################
# ####
# #### 	Feature selection & cross-validation
# ####
# #################################################################
# #################################################################


def feat_selection(dtrain, dheld, infostring):
    x, y, ydict = dtrain.build('m')
    xheld, yheld, _ = dheld.build('m', ydict=ydict)

    #print 'Initial x: {}'.format(x.shape)
    #print 'Initial y: {}'.format(y.shape)
    #print 'Initial xheld: {}'.format(xheld.shape)
    #print 'Initial yheld: {}'.format(yheld.shape)

    #if "noSel" not in infostring:
    #    # always do variance threshold before ANova or RF model selection
    #    # varthresh = 0.3
    #    print 'Thresholding variance at default threshold'
    #    sel = VarianceThreshold()
    #    sel.fit(np.nan_to_num(x))
    #    x = sel.transform(x)
    #    xheld = sel.transform(xheld)
    #    print 'x after variance threshold: {}'.format(x.shape)

    #if 'minmaxNorm' in infostring:
    #    print 'Scaling features to [0,1]'
    #    sel = preprocessing.MinMaxScaler().fit(x)
    #    x = sel.transform(x)
    #    xheld = sel.transform(xheld)

    #if "anovaSel" in infostring:
    #    anovapct = 50
    #    sel = SelectPercentile(f_classif, percentile=anovapct)
    #    sel.fit(x, y)
    #    x = sel.transform(x)
    #    xheld = sel.transform(xheld)
    #    print 'x after filtering: {} features'.format(x.shape)
    #    print 'xheld shape: {}\n'.format(xheld.shape)

    #if "rfSel" in infostring:
    #    print 'Selecting from RF classifier Var importance'
    #    clf = RandomForestClassifier(
    #        n_estimators=150, verbose=False, random_state=1337)
    #    sel = SelectFromModel(clf, threshold='mean')
    #    sel.fit(x, y)
    #    x = sel.transform(x)
    #    xheld = sel.transform(xheld)
    #    print 'x after Feature filtering: {}'.format(x.shape)

    if 'withGleason' in infostring:
        print 'Appending Gleason score to features'
        x = dtrain.append_gleason(x)
        xheld = dheld.append_gleason(xheld)
        print 'x after adding gleason: {} features'.format(x.shape)
        print 'xheld shape: {}\n'.format(xheld.shape)

    return x, y, xheld, yheld


def whole_set_normalization(dataset, infostring):
    x, y, _ = dataset.build('m')

    print 'For normalization, built x: {}'.format(x.shape)
    print 'For normalization, built y: {}'.format(y.shape)

    did_something = False
    if 'minmaxNorm' in infostring:
        print 'Scaling features to [0,1]'
        sel = preprocessing.MinMaxScaler().fit(x)
        x = sel.transform(x)
        did_something = True

    if did_something:
        for i in range(x.shape[0]):
            dataset.data[i].data = x[i, :]

    print 'Dataset.data[0] = {}'.format(dataset.data[0].data.shape)
    return dataset


def hard_selected_features(dataset, infostring):
    n_features = dataset.data[0].data.shape[1]
    if 'rfFeatures' in infostring:
        print 'Using random forest selected features'
        f_use = [
            1, 4, 14, 15, 17, 21, 22, 29, 31, 32, 34, 35, 36, 38, 39, 44, 45,
            50, 56, 57, 58, 59, 61, 62
        ]

    elif 'foldFeatures' in infostring:
        print 'Using fold change based features >|0.3|'
        f_use = [3, 8, 18, 33, 36, 37, 49, 50, 59]

    elif 'noSel' in infostring:
        print 'Returning unchanged feature set'
        return dataset

    else:
        print '\nTypo???????????????? '

    fmask = np.zeros(shape=(1, n_features), dtype=np.bool)
    n_selected = len(f_use)
    for i in f_use:
        fmask[0, i] = True
    for i in range(len(dataset.data)):
        data_selected = dataset.data[i].data[fmask]
        #data_selected = np.swapaxes(data_selected, 0,1)
        data_selected.shape = (n_selected, )  # TODO (nathan) this is dum
        dataset.data[i].data = data_selected

    print 'After filtering, data is {}'.format(dataset.data[0].data.shape)
    return dataset


def run_case_xval(k, dataset, reportfile, cv, infostring, i):
    print '{} ({})'.format(infostring, i)
    print 'Holding out {}'.format(k)

    dtrain, dheld = dataset.hold_out(filterparam='case', skipvalue=k)
    print 'Dataset with {} cases'.format(len(dataset.data))
    print 'Split into {} training'.format(len(dtrain.data))
    print 'Split into {} held-out'.format(len(dheld.data))

    if 'withBalance' in infostring:
        dtrain = balance_dataset(dtrain, 'm')
        dtrain.print_class_pct('m')

    # Feature selection
    # note: 2017-5-4: Moved most of this functionality out
    # Now it creates x and xheld, and appends Gleason grade appropriately
    x, y, xheld, yheld = feat_selection(dtrain, dheld, infostring)

    # build a classifier
    print 'Final x: {}'.format(x.shape)
    print 'Final x_held: {}'.format(xheld.shape)

    # RandomForestClassifier from sklearn
    clf = RandomForestClassifier(n_estimators=150, random_state=1337)

    # cross_val_score from sklearn
    scores = cross_val_score(clf, x, y, cv=cv, n_jobs=4, verbose=False)
    clf.fit(x, y)

    # clf.predict() method from sklearn
    ypred = clf.predict(xheld)

    # Build the report
    raw_acc = (ypred == yheld).sum() / float(yheld.shape[0])
    # print ypred
    # print yheld

    repstr = '{},{},{},{},{}\n'.format(k, yheld[0],
                                       scores.mean(), scores.std() * 2, raw_acc)
    reportfile.write(repstr)

    print repstr
    return raw_acc


#grade_dict = {'3'		: 6,
#			  '3+3'		: 6,
#			  '3+4' 	: 7,
#			  '3+4+5'	: 9,
#			  '3+5'		: 8,
#			  '3,4+5'	: 9,
#			  '4'		: 8,
#			  '4+3'		: 7,
#			  '4+4'		: 8,
#			  '4+5'		: 9,
#			  '5'		: 10,
#			  '5 5'		: 10,
#			  '5+3'		: 8,
#			  '5+4'		: 9,
#			  '5+5'		: 10,
#			  '5_5'		: 10}


def filter_cases(dataset):
    print 'Excluding hard-coded case strings:'
    print 'Before filtering: {}'.format(dataset.print_info())
    dataset.filter_out('case', '03-161')
    dataset.filter_out('case', '08-6659')
    dataset.filter_out('case', '09-7198')
    dataset.filter_out('case', '10-8604')
    dataset.filter_out('case', '11-1997')
    dataset.filter_out('case', '11-2605')
    dataset.filter_out('case', '12-7319')
    dataset.filter_out('case', '12-7463')
    dataset.filter_out('case', '15-7629')

    print 'After filtering: {}'.format(dataset.print_info())
    return dataset


def filter_databy(dataset, infostring):
    if 'scoreLT7' in infostring:
        print 'Filtering cases >=7'
        print 'Before filtering: {}'.format(dataset.print_info())
        #dataset.filter_out('grade', '3+5')
        dataset.filter_out('grade', '5')
        dataset.filter_out('grade', '5 5')
        dataset.filter_out('grade', '5+5')
        dataset.filter_out('grade', '5+3')
        dataset.filter_out('grade', '5_5')
        dataset.filter_out('grade', '5+4')
        dataset.filter_out('grade', '4+5')
        dataset.filter_out('grade', '4+4')
        dataset.filter_out('grade', '3,4+5')
        dataset.filter_out('grade', '3+4+5')

    if 'scoreGT7' in infostring:
        print 'Filtering cases <= 7 and = 3+5'
        print 'Before filtering: {}'.format(dataset.print_info())
        dataset.filter_out('grade', '3+5')
        dataset.filter_out('grade', '4+3')
        dataset.filter_out('grade', '3+4')
        dataset.filter_out('grade', '3')
        dataset.filter_out('grade', '3+3')

    print 'After filtering: {}'.format(dataset.print_info())
    return dataset


def group_databy(dataset, infostring):
    if 'groupbyTile' in infostring:
        print 'Grouping tiles together by averaging'
        dataset = dataset.average_all('label')
    elif 'groupbyCase' in infostring:
        print 'Grouping cases together by averaging'
        dataset = dataset.average_all('case')
    else:
        print 'No recognized grouping to be done'

    return dataset


def parse_args(args):
    '''
    0 = file name
    1 = prepend
    2 = m0
    3 = m1
    4 = infostring
    5 = target results folder
    '''
    prepend, m0, m1, infostring = args[1:]
    m0 = os.path.join(prepend, m0)
    m1 = os.path.join(prepend, m1)

    reportdir = os.path.join(prepend, 'report')
    if not os.path.exists(reportdir):
        os.mkdir(reportdir)

    summaryfile = os.path.join(reportdir, infostring + '.csv')
    if os.path.exists(summaryfile):
        print 'Found existing version of {}'.format(summaryfile)
        os.remove(summaryfile)
        print 'Cleaned it up'

    return m0, m1, summaryfile, infostring


if __name__ == '__main__':

    # do the args:
    m0, m1, summaryfile, infostring = parse_args(sys.argv)
    # print m0, m1, summaryfile, infostring

    ### If I don't want to take in command line args
    # prepend = '/home/nathan/mzmo/data/nuclei/indiv.interior'
    # infostring = 'nuclei_imagenet_fc7_withGleason_anovaSel_noNorm_withBalance'
    # m0 = os.path.join(prepend, 'm0_fc7.p')
    # m1 = os.path.join(prepend, 'm1_fc7.p')

    m0 = populate_tiles(m0, 0, infostring)
    m1 = populate_tiles(m1, 1, infostring)

    data = []
    for k in m0:
        data.append(k)
    for k in m1:
        data.append(k)

    # Instantiate the dataset
    dataset = Dataset(data)

    # Filter out hard-coded cases, if they exist
    dataset = filter_cases(dataset)

    # Remove cases where the 'grade' is just PN or something
    #dataset.filter_out('grade', None)
    dataset.filter_out('grade', 'SC')
    dataset.filter_out('grade', 'NE')

    # If there's custom filtering requested, then do it.
    dataset = filter_databy(dataset, infostring)

    # Group data by some parameters
    dataset = group_databy(dataset, infostring)

    # Perform feature selection on all of the cases together
    dataset = hard_selected_features(dataset, infostring)

    # Whole dataset normalization
    dataset = whole_set_normalization(dataset, infostring)

    # Print out info about the remaining cases to std::out
    cases = dataset.list_param('case')
    print 'Dataset cases: '
    for c in cases:
        print c

    grades = dataset.list_param('grade')
    print 'Dataset grades: '
    for g in grades:
        print g

    # summaryfile = os.path.join(prepend, infostring+'.csv')
    cv = KFold(n_splits=5, shuffle=True, random_state=1337)

    # Run and record results
    with open(summaryfile, 'w') as f:
        f.write('Test case,Class,XVal Mean,XVal Std,TestAccuracy\n')
        acc = [
            run_case_xval(k, dataset, f, cv, infostring, i)
            for i, k in enumerate(cases)
        ]

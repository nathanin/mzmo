#!/home/ingn/miniconda2/bin/python

import pandas as pd
import numpy as np
import glob
import os

# #################################################################
# #################################################################
# ####
# #### 	Function defs
# ####
# #################################################################
# #################################################################


def filenames(searchin):
    pretrained = os.path.join(searchin, '*.csv')
    names = sorted(glob.glob(pretrained))
    for n in names:
        print 'Found {}'.format(n)
    return names


def loadcsv(fpath):
    try:
        data = pd.read_csv(fpath, sep=',', header=1)
    except:
        print 'failed to load {}'.format(fpath)
        data = None
    return data


def x_val_mean(data):
    # col 2
    d = data.values[:, 2]
    return np.mean(d)


def get_sub_data(data, col, value):
    indices = np.where(data.values[:, col] == value)
    return data.values[indices]


def m1_acc(data):
    data = get_sub_data(data, 1, 1)
    d = data[:, 4]
    return np.mean(d)


def thresh_accuracy(data, threshold=[0.1, 0.25, 0.5, 0.75, 1]):
    acc = []
    d = data.values[:, 4]
    n = float(d.shape[0])
    for thr in threshold:
        acc.append((d >= thr).sum() / n)
    return acc


def m0_acc(data):
    data = get_sub_data(data, 1, 0)
    d = data[:, 4]
    return np.mean(d)


def all_acc(data):
    d = data.values[:, 4]
    return np.mean(d)


def n_cases(data):
    return data.values.shape[0]


def header_str():
    return 'File,Grouping,Scores,Features,Normalization,Balancing,Additional,N,N M0,N M1,Cross-Val Mean,M1-Accuracy (mean),M0-Accuracy (mean),All-Accuracy (mean),Thr 10%,Thr 25%,Thr 50%,Thr 75%,Thr 100%'


def n_m0(data):
    return (data.values[:, 1] == 0).sum()


def n_m1(data):
    return (data.values[:, 1] == 1).sum()


def run_analysis(base, data):
    thracc = thresh_accuracy(data)
    retval = '{},{},{},{},{:3.3f},{:3.3f},{:3.3f},{:3.3f}'.format(
        base.replace('_', ','),
        n_cases(data),
        n_m0(data),
        n_m1(data), x_val_mean(data), m1_acc(data), m0_acc(data), all_acc(data))
    for tacc in thracc:
        retval = '{},{:3.3f}'.format(retval, tacc)
    return retval


# #################################################################
# #################################################################
# ####
# #### 	Run
# ####
# #################################################################
# #################################################################

if __name__ == '__main__':

    outfile = '/home/ingn/mzmo/analysis/nuclei/features/summary.csv'
    f = open(outfile, 'w')

    # dirs = [filenames('/home/nathan/mzmo/code/pretrained'),
    # 		filenames('/home/nathan/mzmo/code/sfta_gabor/cases')]
    dirs = [
        filenames(
            '/home/ingn/mzmo/analysis/nuclei/features/report'
        )
    ]
    f.write(header_str() + '\n')

    for names in dirs:
        for n in names:
            base = os.path.basename(n)
            data = loadcsv(n)
            if data is not None:
                retval = run_analysis(base, data)
                f.write(retval + '\n')
                print 'writing {}'.format(retval)

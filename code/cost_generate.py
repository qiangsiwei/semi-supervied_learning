# -*- coding: utf-8 -*-

import re
import os
import json
import glob
import pickle
import random
import numpy as np
import scipy.io.arff as arff

# generate random cost matrix
for filename in glob.glob(r'../dataset/UCI/*.arff'):
    basename = re.sub(r'(\..*?)$','',os.path.basename(filename))
    if basename != 'sick':
        continue
    labels = [row[-1] for row in arff.loadarff(filename)[0]]
    Cmin, Cmax, Cnum = 1, 10, len(set(labels))
    cost_matrix = np.vstack([[random.randint(5,5)]*Cnum for i in xrange(Cnum)])
    # cost_matrix = np.vstack([[round(random.uniform(1.0,1.1),2)]*Cnum for i in xrange(Cnum)])
    print cost_matrix
    basename = re.sub(r'(\..*?)$','',os.path.basename(filename))
    pickle.dump(cost_matrix, open('../dataset/UCI/'+basename+'_cost_matrix.pkl', 'wb'))

# generate cost matrix for data imbalance
for filename in glob.glob(r'../dataset/UCI/*.arff'):
    basename = re.sub(r'(\..*?)$','',os.path.basename(filename))
    if basename != 'sick':
        continue
    labels = [row[-1] for row in arff.loadarff(filename)[0]]
    y = np.array([{v:k for k,v in enumerate(list(set(labels)))}[label] for label in labels])
    Cmin, Cmax, Cnum = 1, 10, len(set(labels))
    print [len(y[y==i]) for i in xrange(Cnum)]
    cost_matrix = np.vstack([[1.0*len(y)/len(y[y==i])/Cnum]*Cnum for i in xrange(Cnum)])
    cost_matrix = np.exp(cost_matrix-cost_matrix.min())
    print cost_matrix
    basename = re.sub(r'(\..*?)$','',os.path.basename(filename))
    pickle.dump(cost_matrix, open('../dataset/UCI/'+basename+'_cost_matrix.pkl', 'wb'))

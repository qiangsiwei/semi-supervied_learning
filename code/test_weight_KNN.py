# -*- coding: utf-8 -*-

import os
import re
import glob
import math
import time
import pickle
import numpy as np
import scipy.io.arff as arff
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Imputer
from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import KFold, train_test_split

DS = 'breast-w'
# DS = 'colic'
# DS = 'diabetes'
# DS = 'ionosphere'
# DS = 'sick'
# DS = 'sonar'
# DS = 'vote'

ALG = {
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'KNN_AB': KNeighborsClassifier(n_neighbors=5),
    'KNN_AB_CS': KNeighborsClassifier(n_neighbors=5)
}

def cross_validation(title, X_label, X_unlabel, y_label, y_unlabel, ones_matrix, cost_matrix):
    #### begin evaluate ####
    accs = {alg:{'acc':[],'cost':[]} for alg in ALG.keys()}
    for alg in sorted(ALG.keys()):
        clf = ALG[alg]
        X_train, y_train = X_label.copy(), y_label.copy()
        X_test, y_test = X_unlabel.copy(), y_unlabel.copy()
        if alg.endswith('KNN'):
            clf.fit(X_train, y_train)
        elif alg.endswith('AB'):
            coefficient_same = np.array([(X_train[i]-X_train[j])**2 for i in xrange(0,len(X_train)-1) for j in xrange(i+1,len(X_train)) if y_train[i]==y_train[j]]).mean(axis=0)
            # coefficient_diff = np.array([(X_train[i]-X_train[j])**2 for i in xrange(0,len(X_train)-1) for j in xrange(i+1,len(X_train)) if y_train[i]!=y_train[j]]).mean(axis=0)
            # print coefficient_same, coefficient_diff
            attribute_weight = np.array([1./k if k!=0 else 0 for k in coefficient_same])
            attribute_weight = attribute_weight/attribute_weight.sum()
            print "%s\t%s\t%s"%(title,alg,'\t'.join(['%.5f'%i for i in list(attribute_weight)]))
            X_train *= attribute_weight
            X_test *= attribute_weight
            clf.fit(X_train, y_train)
        elif alg.endswith('CS'):
            coefficient_same = np.array([((X_train[i]-X_train[j])/cost_matrix[y_train[i]][y_train[j]])**2 for i in xrange(0,len(X_train)-1) for j in xrange(i+1,len(X_train)) if y_train[i]==y_train[j]]).mean(axis=0)
            # coefficient_diff = np.array([((X_train[i]-X_train[j])/cost_matrix[y_train[i]][y_train[j]])**2 for i in xrange(0,len(X_train)-1) for j in xrange(i+1,len(X_train)) if y_train[i]!=y_train[j]]).mean(axis=0)
            # print coefficient_same, coefficient_diff
            attribute_weight = np.array([1./k if k!=0 else 0 for k in coefficient_same])
            attribute_weight = attribute_weight/attribute_weight.sum()
            print "%s\t%s\t%s"%(title,alg,'\t'.join(['%.5f'%i for i in list(attribute_weight)]))
            X_train *= attribute_weight
            X_test *= attribute_weight
            clf.fit(X_train, y_train)
        else:
            continue
        ####### Compute error #######
        # X_test, y_test = X_train, y_train
        error = 1.*sum([1 for a,b in zip(y_test,clf.predict(X_test)) if a!=b])/len(y_test)
        cost = 1.*sum([cost_matrix[a][b] for a,b in zip(y_test,clf.predict(X_test)) if a!=b])/len(y_test)
        accs[alg]['acc'].append(1.-error)
        accs[alg]['cost'].append(cost)
    # for alg in sorted(accs.keys()):
    #     print '\t%.3f %.3f'%(round(np.mean(accs[alg]['acc']),3),round(np.mean(accs[alg]['cost']),3)),
    # print

def test():
    vec = DictVectorizer()
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    for filename in glob.glob(r'../dataset/UCI/*.arff'):
        basename = re.sub(r'(\..*?)$','',os.path.basename(filename))
        print basename
        if basename != DS:
            continue
        # cost_matrix = pickle.load(open('../dataset/UCI/'+basename+'_cost_matrix.pkl', 'rb'))
        data = arff.loadarff(filename)[0]
        X = vec.fit_transform(np.array([{str(i):value for i,value in enumerate(list(row)[:-1])} for row in data])).toarray()
        imp.fit(X)
        X = imp.transform(X)
        labels = np.array([row[-1] for row in data])
        y = np.array([{v:k for k,v in enumerate(list(set(labels)))}[label] for label in labels])
        random = np.random.permutation(range(len(X)))
        print 'dataset ratio\t%s'%('\t'.join([alg+" "*(12-len(alg)) for alg in sorted(ALG.keys())]))
        for iteration in xrange(10):
            X, y, class_num, kf = X[random], y[random], set(labels), KFold(len(X), n_folds=10)
            for train, test in kf:
                length, train_size = len(train), 0.1
                X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
                X_label, X_unlabel, y_label, y_unlabel = train_test_split(X_train, y_train, test_size=1.0-train_size, random_state=0)
                for R in xrange(2,10):
                    ones_matrix, cost_matrix = np.array([[1,1],[1,1]]), np.array([[1,1],[R,R]])            
                    # print "%s R=%d"%(basename,R),
                    cross_validation("%s R=%d"%(basename,R), X_label, X_unlabel, y_label, y_unlabel, ones_matrix, cost_matrix)
                exit()

def show():
    print "not supported yet"

import sys, getopt
opts, args = getopt.getopt(sys.argv[1:], "rsh")
for op, value in opts:
    if op == "-r":
        test()
    elif op == "-s":
        show()
    elif op == "-h":
        print "show usage here!"
        sys.exit()
        
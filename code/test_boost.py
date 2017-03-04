# -*- coding: utf-8 -*-

import os
import re
import glob
import math
import time
import pickle
import numpy as np
import scipy.io.arff as arff
from sklearn.datasets import load_iris
# from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from weight_boosting_semi import AdaBoostClassifier
# from weight_boosting import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import Imputer
from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import KFold, train_test_split
from collections import Counter

DS = 'breast-w'
# DS = 'colic'
# DS = 'diabetes'
# DS = 'ionosphere'
# DS = 'sick'
# DS = 'sonar'
# DS = 'vote'
# DS = 'car'

ALG = {
    'DT': DecisionTreeClassifier(max_depth=3),
    'MlAda_DT': AdaBoostClassifier(n_estimators=10),
    'MlAda_DT_CS': AdaBoostClassifier(n_estimators=10),
    'MlAda_DT_CS_we': AdaBoostClassifier(n_estimators=10),
    'MlAda_DT_CS_al': AdaBoostClassifier(n_estimators=10),
    'SSMAB_DT': AdaBoostClassifier(n_estimators=10),
    'SSMAB_DT_CS': AdaBoostClassifier(n_estimators=10),
    'SSMAB_DT_CS_we': AdaBoostClassifier(n_estimators=10),
    'SSMAB_DT_CS_al': AdaBoostClassifier(n_estimators=10)
}

def cross_validation(X, y, ones_matrix, cost_matrix, cost_weight_matrix):
    # cost_matrix = [[1,1],[1,1]]
    # print cost_matrix
    #### begin evaluate ####
    accs = {alg:{'acc':[],'cost':[]} for alg in ALG.keys()}
    for iteration in xrange(10):
        # print iteration
        random = np.random.permutation(range(len(X)))
        X, y, class_num, kf = X[random], y[random], len(cost_matrix), KFold(len(X), n_folds=10)
        for train, test in kf:
            length, train_size = len(train), 0.1
            X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
            X_label, X_unlabel, y_label, y_unlabel = train_test_split(X_train, y_train, test_size=1.0-train_size, random_state=0)
            y_unlabel = None
            for alg in sorted(ALG.keys()):
                clf = ALG[alg]
                X_train, y_train = X_label, y_label
                if alg.startswith('DT'):
                    clf.fit(X_train, y_train)
                elif alg.startswith('MlAda'):
                    if alg.endswith('CS'):
                        clf.fit(X_train, y_train, cost_matrix=cost_matrix)
                    elif alg.endswith('CS_we'):
                        clf.fit(X_train, y_train, cost_matrix=cost_weight_matrix)
                    elif alg.endswith('CS_al'):
                        clf.fit(X_train, y_train, cost_matrix=cost_matrix, cost_matrix2=ones_matrix)
                    else:
                        clf.fit(X_train, y_train)
                elif alg.startswith('SSMAB_DT'):
                    # alpha为初始权重，beta为加权权重
                    alpha, beta = 1.0, 2.0
                    # ----------------------------------------
                    # 基于KNN给未标记数据打【伪】标签
                    # KNN_time_start = time.time()
                    if y_unlabel == None:
                        clf_knn = KNeighborsClassifier(n_neighbors=5)
                        clf_knn.fit(X_label, y_label)
                        y_unlabel = clf_knn.predict(X_unlabel)
                        # y_unlabel, n_neighbors = np.array([]), 5
                        # for sample_unlabel in X_unlabel:
                        #     dists = np.linalg.norm(X_train-sample_unlabel, axis=1)
                        #     y_unlabel = np.append(y_unlabel, Counter([y_label[sample_index] for sample_index in np.argsort(dists)[:n_neighbors]]).most_common(1)[0][0])
                    # KNN_time_end = time.time()
                    # print KNN_time_end - KNN_time_start
                    # ----------------------------------------
                    # 数据准备
                    X_train = np.vstack([X_label,X_unlabel])
                    y_train = np.concatenate([y_label,y_unlabel])
                    weight_label = np.empty(len(X_label), dtype=np.float)
                    weight_unlabel = np.empty(len(X_unlabel), dtype=np.float)
                    # sample_weight为初始权重
                    weight_label[:] = alpha*1.0/(length*(1+(alpha-1)*train_size))
                    weight_unlabel[:] = 1.0/(length*(1+(alpha-1)*train_size))
                    sample_weight = np.concatenate([weight_label,weight_unlabel])
                    # sample_weight为加权权重
                    weight_label[:] = beta
                    weight_unlabel[:] = 1.0
                    label_weight = np.concatenate([weight_label,weight_unlabel])
                    if alg.endswith('CS'):
                        clf.fit(X_train, y_train, sample_weight=sample_weight, label_weight=label_weight, cost_matrix=cost_matrix)
                    elif alg.endswith('CS_we'):
                        clf.fit(X_train, y_train, sample_weight=sample_weight, label_weight=label_weight, cost_matrix=cost_weight_matrix)
                    elif alg.endswith('CS_al'):
                        clf.fit(X_train, y_train, sample_weight=sample_weight, label_weight=label_weight, cost_matrix=cost_matrix, cost_matrix2=ones_matrix)
                    else:
                        clf.fit(X_train, y_train, sample_weight=sample_weight, label_weight=label_weight)
                else:
                    continue
                ####### Compute error #######
                # X_test, y_test = X_train, y_train
                error = 1.*sum([1 for a,b in zip(y_test,clf.predict(X_test)) if a!=b])/len(y_test)
                cost = 1.*sum([cost_matrix[a][b] for a,b in zip(y_test,clf.predict(X_test)) if a!=b])/len(y_test)
                accs[alg]['acc'].append(1.-error)
                accs[alg]['cost'].append(cost)
    for alg in sorted(accs.keys()):
        print '\t%.3f %.3f'%(round(np.mean(accs[alg]['acc']),3),round(np.mean(accs[alg]['cost']),3)),
        # print '\t%s %.3f %.3f'%(alg,round(np.mean(accs[alg]['acc']),3),round(np.mean(accs[alg]['cost']),3)),
    print

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
        # print {v:k for k,v in enumerate(list(set(labels)))}
        # y = np.array([{v:k for k,v in enumerate(list(set(labels)))}[label] for label in labels])
        y = np.array([{'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3}[label] for label in labels]) #car
        print 'dataset ratio\t%s'%('\t'.join([alg+" "*(12-len(alg)) for alg in sorted(ALG.keys())]))
        for R in xrange(2,10):
            time_start = time.time()
            # 测试不同代价矩阵
            # ones_matrix, cost_matrix = np.array([[1,1],[1,1]]), np.array([[1,1],[R,R]])
            ones_matrix, cost_matrix = np.array([[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]]), np.array([[1]*4,[1+1*R]*4,[1+2*R]*4,[1+3*R]*4])
            # 加权同时优化准确率和总代价
            lambda_weight = 1.
            cost_matrix_weighted = ones_matrix*(1./(1.+lambda_weight))+cost_matrix*(lambda_weight/(1.+lambda_weight))
            print "%s R=%d"%(basename,R),
            cross_validation(X, y, ones_matrix, cost_matrix, cost_matrix_weighted)
            time_end = time.time()
            print "time consumption: %f"%(time_end - time_start)

def show():
    p = re.compile(r'^(.*?)\s\t(.*?)\s\t(.*?)\s\t(.*?)\s\t(.*?)\s\t(.*?)\s\t(.*?)\s\t(.*?)\s\t(.*?)\s\t(.*?)$')
    f = open('../docs/result.txt','r')
    data = [p.findall(line)[0] for line in f.read().strip().split('\n')[1:]]
    for line in data:
        costs = [float(d.split(' ')[1]) for d in line[1:]]
        print line[0], sorted(ALG)[costs.index(min(costs))]
        # print '\t'.join(line), '\t%.3f'%(1.-float(line[3].split(' ')[-1])/float(line[2].split(' ')[-1])), '\t%.3f'%(1.-float(line[5].split(' ')[-1])/float(line[4].split(' ')[-1])), '\t%.3f'%(1.-float(line[5].split(' ')[-1])/float(line[3].split(' ')[-1])) 

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

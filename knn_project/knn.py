#!/usr/bin/python3

from numpy import *
import operator


def createdataset():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inx, dataset, labels, k):
    datasetsize = dataset.shape[0]
    diffmat = tile(inx, (datasetsize, 1)) - dataset
    sqdiffmat = diffmat ** 2
    sqdistances = sqdiffmat.sum(axis=1)
    distances = sqdistances ** 0.5
    sortedDistIndicies = distances.agsort()
    classcount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classcount[voteIlabel] = classcount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classcount.items(),
                    key=operator.itemgetter(1), reverse=True)
    return sortedDistIndicies[0][0]




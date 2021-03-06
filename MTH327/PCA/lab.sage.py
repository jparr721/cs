
# This file was *autogenerated* from the file lab.sage
from sage.all_cmdline import *   # import sage library

_sage_const_3 = Integer(3); _sage_const_2 = Integer(2); _sage_const_1 = Integer(1); _sage_const_0 = Integer(0); _sage_const_4 = Integer(4); _sage_const_30 = Integer(30); _sage_const_50 = Integer(50)
import csv
import numpy as np

def findmean(data):
    return vector(np.sum(data, axis=_sage_const_0 ))/len(data)

def demean(data):
    mean = findmean(data)
    return matrix(RDF, [datum - mean for datum in data]).T

def plot2d(M):
    colors = ['red', 'green', 'blue']
    p = list_plot([])
    for i in range(_sage_const_3 ):
        p += list_plot(M.columns()[_sage_const_50 *i: _sage_const_50 *(i+_sage_const_1 )], color = colors[i], aspect_ratio=_sage_const_1 , size=_sage_const_30 )
    return p


input = csv.reader(open('iris.data'))
data = map(vector, [map(float, datum[:_sage_const_4 ]) for datum in input])

# Average petal length in cm
print findmean(data)[_sage_const_2 ]


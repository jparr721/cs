import csv
import numpy as np

def findmean(data):
    return vector(np.sum(data, axis=0))/len(data)

def demean(data):
    mean = findmean(data)
    return matrix(RDF, [datum - mean for datum in data]).T

def plot2d(M):
    colors = ['red', 'green', 'blue']
    p = list_plot([])
    for i in range(3):
        p += list_plot(M.columns()[50*i: 50*(i+1)], color = colors[i], aspect_ratio=1, size=30)
    return p


input = csv.reader(open('iris.data'))
data = map(vector, [map(float, datum[:4]) for datum in input])

# Average petal length in cm
print findmean(data)[2]

# Construct the covariance matrix C
A = demean(data)
C = 1/len(data) * A * A.T

# Then, we can use the eigenvalues to begin constructing the variance
C.eigenvalues()

# Extract our principal components
sum(C.eigenvalues()[:2])/sum(C.eigenvalues() * 100)

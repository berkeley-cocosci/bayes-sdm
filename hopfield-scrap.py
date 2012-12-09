import numpy as np
import matplotlib.pyplot as plt
from util import corrupt, plot_io
import hopfield as hopfield

def like_matlab(arr):
    n = int(np.sqrt(arr.shape[0]))
    newarr = arr[:, 0].reshape((n, n), order='F').ravel()[:, None]
    return newarr

data = np.load('patterns.npz')
X = data['X']
hi = data['hi']
face = data['face']
num1 = data['num1']
num2 = data['num2']
num3 = data['num3']
num4 = data['num4']
data.close()
#inputs = np.hstack([face, X, hi, num1, num2, num3, num4])
inputs = np.hstack([face, X, hi, num1, num2])

hop = hopfield.hopnet(inputs)
addresses = inputs.copy()
numNeurons, numPatterns = addresses.shape

numIters = 1000

print "uncorrupted tests"
for i in xrange(numPatterns):
    a = addresses[:, i]
    d = hop.read(a, numIters).reshape((10, 10), order='F')
    plt.figure(i)
    plot_io(a.reshape((10, 10), order='F'), d)

print "now with corruption"
for i in xrange(numPatterns):
    a = corrupt(addresses[:, i], 10)
    d = hop.read(a, numIters).reshape((10, 10), order='F')
    plt.figure(i + numPatterns)
    plot_io(a.reshape((10, 10), order='F'), d)

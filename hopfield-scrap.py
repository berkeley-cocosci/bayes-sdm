import numpy as np
import matplotlib.pyplot as plt
from util import corrupt, plot_io
import hopfield as hopfield

def like_matlab(arr):
    n = int(np.sqrt(arr.shape[0]))
    newarr = arr[:, 0].reshape((n, n), order='F').ravel()[:, None]
    return newarr

data = np.load('patterns.npz')
X = like_matlab(data['X'])
hi = like_matlab(data['hi'])
face = like_matlab(data['face'])
data.close()
num1 = like_matlab(np.load('1.npy'))
num2 = like_matlab(np.load('2.npy'))
num3 = like_matlab(np.load('3.npy'))
num4 = like_matlab(np.load('4.npy'))
#inputs = np.hstack([face, X, hi, num1, num2, num3, num4])

inputs = np.hstack([face, X, hi, num1, num2]).astype('f8')

hop = hopfield.hopnet(inputs)
addresses = inputs.copy()
numNeurons,numPatterns = addresses.shape

numIters = 1000



print "uncorrupted tests"
for i in xrange(numPatterns):
    a = addresses[:, i]
    d = hop.read(numIters,a).reshape((10, 10), order='F')
    plt.figure(i)
    plot_io(a.reshape((10, 10), order='F'), d)





print "now with corruption"

for i in xrange(numPatterns):
    a = corrupt(addresses[:, i], 10)
    d = hop.read(numIters,a).reshape((10, 10), order='F')
    plt.figure(i+numPatterns)
    plot_io(a.reshape((10, 10), order='F'), d)

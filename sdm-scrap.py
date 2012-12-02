# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import sdm as sdm
from util import corrupt, plot_io

# <codecell>

def like_matlab(arr):
    n = int(np.sqrt(arr.shape[0]))
    newarr = arr[:, 0].reshape((n, n), order='F').ravel()[:, None]
    return newarr

# <codecell>

data = np.load('patterns.npz')
X = like_matlab(data['X'])
hi = like_matlab(data['hi'])
face = like_matlab(data['face'])
data.close()
num1 = like_matlab(np.load('1.npy'))
num2 = like_matlab(np.load('2.npy'))
num3 = like_matlab(np.load('3.npy'))
num4 = like_matlab(np.load('4.npy'))
inputs = np.hstack([face, X, hi, num1, num2, num3, num4])

# <codecell>

reload(sdm)
mem = sdm.SDM(100, 10000, 40)
print "Addresses in hamming radius:", mem._select(inputs).sum(axis=0)
addresses = inputs.copy()
mem.writeM(addresses, inputs)

# <codecell>

for i in xrange(7):
    a = corrupt(addresses[:, i], 10)
    d = mem.read(a).reshape((10, 10), order='F')
    plt.figure(i)
    plot_io(a.reshape((10, 10), order='F'), d)

# <codecell>

reload(sdm)
b = sdm.f2b(2.000001)
print b
f = sdm.b2f(b)
print f

# <codecell>

c = 5 # amount of bits to corrupt
n = 15 # number of exemplars

exemplars = np.empty((100, n))
plt.figure(1)
plt.clf()
for i in xrange(n):
    e = corrupt(face, c)
    exemplars[:, i] = e[:, 0]
    plt.subplot(3, 5, i+1)
    plt.imshow(e.reshape((10, 10), order='F'), cmap='gray', interpolation='nearest')
    plt.xticks([], [])
    plt.yticks([], [])

    

# <codecell>

reload(sdm)
mem = sdm.SDM(100, 10000, 40)
addresses = exemplars.copy()
mem.writeM(addresses, exemplars)

ex = corrupt(face, c)
for i in xrange(1):
    proto = mem.read(ex)
    plt.figure()
    plot_io(ex.reshape((10, 10), order='F'), proto.reshape((10, 10), order='F'))
    ex = proto.copy()

# <codecell>

inputs.shape


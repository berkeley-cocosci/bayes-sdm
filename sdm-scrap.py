# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import matplotlib.pyplot as plt
import sdm as sdm
from util import corrupt, plot_io

# <codecell>

data = np.load('patterns.npz')
X = data['X']
hi = data['hi']
face = data['face']
num1 = data['num1']
num2 = data['num2']
num3 = data['num3']
num4 = data['num4']
data.close()
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

c = 5 # amount of bits to corrupt
n = 15 # number of exemplars

exemplars = np.empty((100, n), dtype='i4')
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
    proto = mem.readM(ex)
    plt.figure()
    plot_io(ex.reshape((10, 10), order='F'), proto.reshape((10, 10), order='F'))
    ex = proto.copy()


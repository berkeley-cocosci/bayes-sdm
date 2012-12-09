# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=1>

# SDM/Hopfield Comparison Code

# <codecell>

"""SDM/Hopfield Comparison

This code compares SDM networks and Hopfield networks on various
properties, including appropriate parameter values, storage capacity,
"prototype" retrieval, and sequence retrieval.

Authors: 
  Jessica Hamrick (jhamrick@berkeley.edu)
  Josh Abbott (joshua.abbott@berkeley.edu)

"""

# <codecell>

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import sdm as sdm
import hopfield as hop
import util as util
import metrics as metrics

# <headingcell level=1>

# Parameters

# <codecell>

# SDM Parameters
# --------------
#   n : length of inputs
#   m : number of addresses
#   D : hamming radius


# Hopfield Parameters
# -------------------
#   n : length of inputs

# <codecell>

# length of inputs
n = 100

# hamming distance encompasses 2.5% of addresses
D = float((n / 2.) - (np.sqrt(n*(0.5**2)) * 1.96))
M = np.arange(200, 1+2000, 200)

# error thresholds
thresh = 0.0

# noise
noise = np.array([0.0, 0.01, 0.02, 0.04, 0.08, 0.16])

# number of simulations to run
iters = 100

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

# <headingcell level=1>

# Uncorrupted Storage Capacity

# <codecell>

# How many random, uncorrelated inputs can the SDM store?
sdm_capacity = np.empty(M.size)
k = 1
for midx, m in enumerate(M):
    k = metrics.test_capacity(
	(int(n), int(m), float(D)), 
	k=int(k), iters=int(iters), 
	thresh=thresh, verbose=False) - 1
    sdm_capacity[midx] = k
    print "SDM (m=%d) capacity is %d" % (m, k)

# <codecell>

# How many random, uncorrelated inputs can the Hopfield net store?
k = metrics.test_capacity(
    int(n), k=1, iters=int(iters), 
    thresh=thresh, verbose=False) - 1
hop_capacity = k
print "Hopfield capacity is %d" % k

# <codecell>

# plot the storage capacity as a function of address space size
util.set_fig_properties()
data = np.hstack([hop_capacity, sdm_capacity])
x = np.arange(data.size)
labels = np.hstack(["Hop.", M])

plt.bar(x[:1], data[:1], align='center', color='r')
plt.bar(x[1:], data[1:], align='center', color='b')

plt.xticks(x, labels)
plt.xlim(-1, data.size)
plt.xlabel("M (# addresses)")
plt.ylabel("Capacity (# uncorrupted items)")
plt.title("SDM and Hopfield Capacities (N=%d)" % n)
plt.legend(loc=0)

# <codecell>

# plot the utilization as a function of address space size
util.set_fig_properties()
data = np.hstack([100*hop_capacity/float(n), 100*sdm_capacity/M.astype('f8')])
x = np.arange(data.size)
labels = np.hstack(["Hop.", M])

plt.bar(x[:1], data[:1], align='center', color='r')
plt.bar(x[1:], data[1:], align='center', color='b')

plt.xticks(x, labels)
plt.xlim(-1, data.size)
plt.xlabel("M (# addresses)")
plt.ylabel("Percent Utilization")
plt.title("SDM and Hopfield Utilizations (N=%d)" % n)
plt.legend(loc=0)

# <headingcell level=1>

# Noise Tolerance

# <codecell>

# How many random, uncorrelated inputs can the SDM store and be able
# to retrieve even with corruption?
M = np.arange(100, 1+1000, 100)
sdm_tolerance = np.empty((noise.size, M.size))
for nidx, err in enumerate(noise):
    k = 1
    for midx, m in enumerate(M):
	k = metrics.test_noise_tolerance(
	    (int(n), int(m), float(D)), 
	    k=int(k), noise=float(err),
	    iters=int(iters), thresh=thresh, verbose=False) - 1
	sdm_tolerance[nidx, midx] = k
	print "m=%d : capacity is %d (%d%% corruption)" % (m, k, err*100)

# <codecell>

# How many random, uncorrelated inputs can the Hopfield net store?
hop_tolerance = np.empty(noise.size)
k = 1
for nidx, err in enumerate(noise):
    k = metrics.test_noise_tolerance(
	int(n), k=1, noise=float(err),
	iters=int(iters), thresh=thresh, verbose=False) - 1
    hop_tolerance[nidx] = k
    print "capacity is %d (%d%% error corruption)" % (k, err*100)

# <codecell>

# plot the storage capacity as a function of address space size
util.set_fig_properties()
data = np.hstack([hop_tolerance[:, None], sdm_tolerance])
x = np.arange(data.shape[1])
labels = np.hstack(["Hop.", M])

for nidx, err in enumerate(noise):
    f = (nidx+1.) / len(noise)
    scolor = (f, f, f)
    hcolor = (1, f, f)
    plt.bar(x[:1], data[nidx, :1], align='center', color=hcolor)
    plt.bar(x[1:], data[nidx, 1:], align='center', color=scolor)

plt.xticks(x, labels)
plt.xlim(-1, x.size)
plt.xlabel("M (# addresses)")
plt.ylabel("Capacity (# uncorrupted items)")
plt.title("SDM and Hopfield Error Tolerance (N=%d)" % n)
plt.legend(loc=0)


# <headingcell level=1>

# Prototype Retrieval

# <codecell>


# <headingcell level=1>

# Sequence Retrieval

# <codecell>



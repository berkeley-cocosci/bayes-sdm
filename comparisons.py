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

# error thresholds
thresh = np.array([0, 0.01, 0.025, 0.05])

# number of simulations to run
iters = 100

# <headingcell level=1>

# Uncorrupted Storage Capacity

# <codecell>

# How many random, uncorrelated inputs can the SDM store?
M = np.arange(100, 1+1000, 100)
sdm_capacity = np.empty((thresh.size, M.size))
for tidx, t in enumerate(thresh):
    k = 1
    for midx, m in enumerate(M):
	k = metrics.test_capacity(
	    (int(n), int(m), float(D)), 
	    k=int(k), iters=int(iters), 
	    thresh=float(t), verbose=False)
	sdm_capacity[tidx, midx] = k-1
	#print "m=%d : capacity is %d (%d%% error tolerance)" % (m, k-1, t*100)

# <codecell>

# How many random, uncorrelated inputs can the Hopfield net store?
hop_capacity = np.empty(thresh.size)
k = 1
for tidx, t in enumerate(thresh):
    k = metrics.test_capacity(
	int(n), k=int(k), iters=int(iters), 
	thresh=float(t), verbose=False)
    hop_capacity[tidx] = k-1
    #print "capacity is %d (%d%% error tolerance)" % (k-1, t*100)

# <codecell>

# plot the storage capacity as a function of address space size
plt.clf()
colors = ['b', 'g', 'r', 'c']
for tidx, t in enumerate(thresh):
    plt.plot(M, hop_capacity[tidx, None]*np.ones_like(M), 
	     label='Hopfield %.1f%% err. tol.' % (t*100), 
	     color=colors[tidx], linestyle='--')
    plt.plot(M, sdm_capacity[tidx], 
	     label='SDM %.1f%% err. tol.' % (t*100), 
	     color=colors[tidx], linestyle='-')
plt.xlabel("M (# addresses)")
plt.ylabel("Capacity (# uncorrupted items)")
plt.title("SDM and Hopfield Capacities (N=%d)" % n)
plt.legend(loc=0)

# <codecell>

# plot the utilization as a function of address space size
plt.clf()
for tidx, t in enumerate(thresh):
    plt.plot(M, 100*hop_capacity[tidx, None]*np.ones_like(M) / float(n), 
	     label='Hopfield %.1f%% err. tol.' % (t*100),
	     color=colors[tidx], linestyle='--')
    plt.plot(M, 100 * sdm_capacity[tidx] / M.astype('f8'),
	     label='SDM %.1f%% err. tol.' % (t*100),
	     color=colors[tidx], linestyle='-')
plt.xlabel("M (# addresses)")
plt.ylabel("Percent Utilization")
plt.title("SDM and Hopfield Utilizations (N=%d)" % n)
plt.legend(loc=0)

# <headingcell level=1>

# Noise Tolerance

# <codecell>

# How many random, uncorrelated inputs can the SDM store?
M = [100, 200, 400, 600, 800, 1000]
capacity = []
for m in M:
    rso.seed(0)
    mem = sdm.SDM(n, m, D, seed=rso)
    corruption = test_uncorrelated_capacity(mem, iters=100, rso=rso, verbose=False)
    capacity.append(corruption.shape[0]-1)
    print "m=%d : capacity is %d" % (M, capacity[-1])

# <headingcell level=1>

# Prototype Retrieval

# <codecell>


# <headingcell level=1>

# Sequence Retrieval

# <codecell>



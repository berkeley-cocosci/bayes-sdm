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
#import hopfield as hopfield
import util as util

# <codecell>

rso = np.random.RandomState(0)

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
D = (n / 2.) - (np.sqrt(n*(0.5**2)) * 1.96) 

# <headingcell level=1>

# Uncorrupted Storage Capacity

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

# <codecell>

# plot the storage capacity as a function of address space size
plt.clf()
plt.plot(M, capacity)
plt.xticks(M, M)
plt.xlabel("# addresses")
plt.ylabel("# items")
plt.title("Number of items stored in the SDM without corruption")

# <codecell>

# How many random, uncorrelated inputs can the Hopfield net store?

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


# <headingcell level=1>

# Address/Data Distinction

# <codecell>



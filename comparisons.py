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

# print extra information
verbose = True

# length of inputs
n = 16**2
# number of simulations to run
iters = 10
# step by which to test storage capacity
k = sorted(set(np.round(
    np.logspace(
	np.log(5)/np.log(10),
	np.log(200)/np.log(10),
	num=20, base=10),
    decimals=0).astype('i8')))

# hamming distance encompasses 2.5% of addresses
D = float((n / 2.) - (np.sqrt(n*(0.5**2)) * 1.96))
#D = 105
# address spaces for the SDM
M = np.array([500, 1000, 2500, 5000, 10000])

# corruption levels to test
noise = np.array([0.0, 0.025, 0.05, 0.10, 0.15, 0.20])

# colors for plotting
colors = ['#FF0000', # red 
	  '#FF9933', # orange
	  '#FFFF00', # yellow
	  '#00FF00', # green
	  '#00FFFF', # cyan
	  '#0000FF', # blue
	  '#660099', # violet
	  # '#FF00FF', # magenta
	  '#FF0099', # pink
	  ]
# alpha value for error regions in plots
alpha = 0.3

# <headingcell level=1>

# Uncorrupted Storage Capacity

# <codecell>

# How many random, uncorrelated inputs can the SDM store?
sdm_capacity = np.empty((len(M), len(k), 2))
for midx, m in enumerate(M):
    print "SDM (M=%d)" % m
    sdm_capacity[midx] = metrics.test_capacity(
	(int(n), int(m), float(D)), 
	k0=k, iters=int(iters), verbose=verbose)

# <codecell>

# How many random, uncorrelated inputs can the Hopfield net store?
print "Hopfield"
hop_capacity = metrics.test_capacity(
    int(n), k0=k, iters=int(iters), verbose=verbose)

# <codecell>

# plot corruption as a function of items stored
util.set_fig_properties()
for i in xrange(M.size):
    util.plot_error(k, sdm_capacity[i], colors[i], alpha, 'SDM M=%d' % M[i])
util.plot_error(k, hop_capacity, colors[i+1], alpha, 'Hopfield', linestyle='--')

plt.xlim(k[0], k[-1])
plt.xlabel("Number of stored items")
plt.ylabel("Mean fraction of corrupted bits")
plt.title("SDM and Hopfield Capacities for Ideal Inputs (N=%d)" % n)
plt.legend(loc=0)

# <headingcell level=1>

# Noise Tolerance

# <codecell>

# How many random, uncorrelated inputs can the SDM store and be able
# to retrieve even with corruption?
m = M[-1]
sdm_tolerance = np.empty((len(noise), len(k), 2))
for nidx, err in enumerate(noise):
    print "SDM (M=%d) w/ %.2f%% corruption" % (m, err*100)
    sdm_tolerance[nidx] = metrics.test_noise_tolerance(
	(int(n), int(m), float(D)), 
	k0=k, noise=float(err),
	iters=int(iters), verbose=verbose)
    

# <codecell>

# How many random, uncorrelated inputs can the Hopfield net store and
# be able to retrieve even with corruption?
hop_tolerance = np.empty((len(noise), len(k), 2))
for nidx, err in enumerate(noise):
    print "Hopfield w/ %.2f%% corruption" % (err*100)
    hop_tolerance[nidx] = metrics.test_noise_tolerance(
	int(n), k0=k, noise=float(err),
	iters=int(iters), verbose=verbose)

# <codecell>

# plot the storage capacity as a function of address space size
util.set_fig_properties()

for nidx, err in enumerate(noise):
    util.plot_error(k, sdm_tolerance[nidx], colors[nidx], alpha,
		    "SDM %d%% corruption" % (err*100),
		    linestyle='-')
    util.plot_error(k, hop_tolerance[nidx], colors[nidx], alpha,
		    "Hop %d%% corruption" % (err*100),
		    linestyle='--')

plt.xlim(k[0], k[-1])
plt.xlabel("Number of stored items")
plt.ylabel("Fraction of corrupted bits")
plt.title("SDM (M=%d) and Hopfield Error Tolerance (N=%d)" % (m, n))
plt.legend(loc=0)


# <headingcell level=1>

# Prototype Retrieval

# <codecell>

# How well can the SDM/Hopfield net store prototypes?
m = M[-1]
kp = 3
prototype_noise = noise.copy()
prototype_k = np.array([10, 25, 50, 100])

sdm_prototype = np.empty((prototype_noise.size, prototype_k.size))
hop_prototype = np.empty((prototype_noise.size, prototype_k.size))
for nidx, err in enumerate(prototype_noise):
    print "Noise:", err
    for kidx, k in enumerate(prototype_k):
	print "SDM",
	p = metrics.test_prototype(
	    (int(n), int(m), float(D)), 
	    kp=kp, ke=int(k), noise=float(err),
	    iters=int(iters), verbose=verbose)
	sdm_prototype[nidx, kidx] = p
	print "Hop",
	p = metrics.test_prototype(
	    int(n), kp=kp, ke=int(k), noise=float(err),
	    iters=int(iters), verbose=verbose)
	hop_prototype[nidx, kidx] = p

	print

# <codecell>

plt.clf()

plt.subplot(1, 3, 1)
plt.imshow(
    1-sdm_prototype, 
    cmap='gray', interpolation='nearest', vmin=0.5, vmax=1)
plt.xticks(np.arange(prototype_k.size), prototype_k)
plt.yticks(np.arange(noise.size), noise)
plt.title("SDM (M=%d)" % m)
plt.xlabel("Exemplars")
plt.ylabel("Percent Corruption")

plt.subplot(1, 3, 2)
plt.imshow(
    1-hop_prototype, 
    cmap='gray', interpolation='nearest', vmin=0.5, vmax=1)
plt.xticks(np.arange(prototype_k.size), prototype_k)
plt.yticks(np.arange(noise.size), [])
plt.title("Hopfield")
plt.xlabel("Exemplars")

plt.subplot(1, 3, 3)
diff = hop_prototype-sdm_prototype
plt.imshow(
    np.sign(diff)*(2**np.log(np.abs(diff))), 
    #diff,
    cmap='RdBu', interpolation='nearest', vmin=-0.5, vmax=0.5)
plt.xticks(np.arange(prototype_k.size), prototype_k)
plt.yticks(np.arange(noise.size), [])
plt.title("Hopfield-SDM")
plt.xlabel("Exemplars")


# <headingcell level=1>

# Sequence Retrieval

# <codecell>


# <codecell>

# # plot the storage capacity as a function of address space size
# util.set_fig_properties()
# data = np.hstack([hop_capacity, sdm_capacity])
# x = np.arange(data.size)
# labels = np.hstack(["Hop.", M.astype('str')])

# plt.bar(x[:1], data[:1], align='center', color='r')
# plt.bar(x[1:], data[1:], align='center', color='b')

# plt.xticks(x, labels)
# plt.xlim(-1, data.size)
# plt.xlabel("M (# addresses)")
# plt.ylabel("Capacity (# uncorrupted items)")
# plt.title("SDM and Hopfield Capacities (N=%d)" % n)
# plt.legend(loc=0)

# <codecell>

# # plot the utilization as a function of address space size
# util.set_fig_properties()
# data = np.hstack([100*hop_capacity/float(n), 100*sdm_capacity/M.astype('f8')])
# x = np.arange(data.size)
# labels = np.hstack(["Hop.", M.astype('str')])

# plt.bar(x[:1], data[:1], align='center', color='r')
# plt.bar(x[1:], data[1:], align='center', color='b')

# plt.xticks(x, labels)
# plt.xlim(-1, data.size)
# plt.xlabel("M (# addresses)")
# plt.ylabel("Percent Utilization")
# plt.title("SDM and Hopfield Utilizations (N=%d)" % n)
# plt.legend(loc=0)


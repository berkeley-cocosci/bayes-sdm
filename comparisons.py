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
n = 16**2

# hamming distance encompasses 2.5% of addresses
D = float((n / 2.) - (np.sqrt(n*(0.5**2)) * 1.96))
#M = np.arange(200, 1+2000, 200)
M = np.array([400, 800, 1600, 3200, 6400, 12800])

# error thresholds
thresh = 0.005
# noise
noise = np.array([0.0, 0.01, 0.02, 0.04, 0.08, 0.16])
# number of simulations to run
iters = 20

k0 = 10

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

verbose = True

# <headingcell level=1>

# Uncorrupted Storage Capacity

# <codecell>

# How many random, uncorrelated inputs can the SDM store?
sdm_capacity = []
for midx, m in enumerate(M):
    print "SDM (M=%d)" % m
    data = metrics.test_capacity(
	(int(n), int(m), float(D)), 
	k0=k0, iters=int(iters), 
	thresh=thresh, verbose=verbose)
    sdm_capacity.append(data)

# <codecell>

# How many random, uncorrelated inputs can the Hopfield net store?
print "Hopfield"
hop_capacity = metrics.test_capacity(
    int(n), k0=k0, iters=int(iters), 
    thresh=thresh, verbose=verbose)

# <codecell>

# plot corruption as a function of items stored
util.set_fig_properties()
fig = plt.gcf()
fig.set_figwidth(8)
fig.set_figheight(6)
alpha = 0.3

data = hop_capacity
loerr = data[:,0]-data[:,1]
hierr = data[:,0]+data[:,1]
x = k0*np.arange(1, data.shape[0]+1)
plt.fill_between(x, loerr, hierr, color=colors[0], alpha=alpha)
plt.plot(x, data[:,0], label='Hopfield', color=colors[0])

for i in xrange(M.size):
    data = sdm_capacity[i]
    loerr = data[:,0]-data[:,1]
    hierr = data[:,0]+data[:,1]
    x = k0*np.arange(1, data.shape[0]+1)
    plt.fill_between(x, loerr, hierr, color=colors[i+1], alpha=alpha)
    plt.plot(x, data[:,0], label='M=%d' % M[i], color=colors[i+1])

plt.ylim(0, thresh)
plt.xlabel("Number of stored items")
plt.ylabel("Mean fraction of corrupted bits")
plt.title("SDM and Hopfield Capacities for Ideal Inputs (N=%d)" % n)
plt.legend(loc=0)

# <headingcell level=1>

# Noise Tolerance

# <codecell>

# How many random, uncorrelated inputs can the SDM store and be able
# to retrieve even with corruption?
m = 12800
sdm_tolerance = []
hop_tolerance = []
for nidx, err in enumerate(noise):
    print "SDM (M=%d) w/ %f%% corruption" % (m, err*100)
    sdm_tolerance.append(metrics.test_noise_tolerance(
	(int(n), int(m), float(D)), 
	k0=k0, noise=float(err),
	iters=int(iters), thresh=thresh, verbose=verbose))
    print "Hopfield w/ %f%% corruption" % (err*100)
    hop_tolerance.append(metrics.test_noise_tolerance(
	int(n), k0=k0, noise=float(err),
	iters=int(iters), thresh=thresh, verbose=verbose))

# <codecell>

# plot the storage capacity as a function of address space size
util.set_fig_properties()
fig = plt.gcf()
fig.set_figwidth(10)
fig.set_figheight(8)

for nidx, err in enumerate(noise):
    data = hop_tolerance[nidx]
    x = k0*np.arange(1, data.shape[0]+1)
    loerr = data[:, 0] - data[:, 1]
    hierr = data[:, 0] + data[:, 1]
    plt.fill_between(x, loerr, hierr, color=colors[nidx], alpha=alpha)
    plt.plot(x, data[:, 0], color=colors[nidx], linestyle='--', 
	     label="Hop %d%% corruption" % (err*100))

    data = sdm_tolerance[nidx]
    x = k0*np.arange(1, data.shape[0]+1)
    loerr = data[:, 0] - data[:, 1]
    hierr = data[:, 0] + data[:, 1]
    plt.fill_between(x, loerr, hierr, color=colors[nidx], alpha=alpha)
    plt.plot(x, data[:, 0], color=colors[nidx], linestyle='-', 
	     label="SDM %d%% corruption" % (err*100))

plt.xlim(5, 135)
plt.ylim(0, thresh)
plt.xlabel("Number of stored items")
plt.ylabel("Fraction of corrupted bits")
plt.title("SDM (M=%d) and Hopfield Error Tolerance (N=%d)" % (m, n))
plt.legend(loc=0)


# <headingcell level=1>

# Prototype Retrieval

# <codecell>

reload(metrics)

# How well can the SDM/Hopfield net store prototypes?
m = 10000
kp = 10
iters = 100
prototype_noise = np.arange(.05, 0.45, 0.05)
prototype_k = np.array([5, 10, 20])
sdm_prototype = np.empty(
	(prototype_noise.size, prototype_k.size))
hop_prototype = np.empty(
	(prototype_noise.size, prototype_k.size))
for nidx, err in enumerate(prototype_noise):
    for kidx, k in enumerate(prototype_k):
	p = metrics.test_prototype(
	    (int(n), int(m), float(D)), 
	    kp=kp, ke=int(k), noise=float(err),
	    iters=int(iters), verbose=verbose)
	sdm_prototype[nidx, kidx] = p
	print "%2d ex.: SDM prototype corruption is %f (%d%% corruption)" % (k, p, err*100)
	p = metrics.test_prototype(
	    int(n), kp=kp, ke=int(k), noise=float(err),
	    iters=int(iters), verbose=verbose)
	hop_prototype[nidx, kidx] = p
	print "%2d ex.: Hopfield prototype corruption is %f (%d%% corruption)" % (k, p, err*100)
    print

# <codecell>

 plt.clf()

plt.subplot(1, 3, 1)
plt.imshow(
    1-sdm_prototype, 
    cmap='gray', interpolation='nearest', vmin=0.5, vmax=1)
plt.xticks(np.arange(prototype_k.size), prototype_k)
plt.yticks(np.arange(prototype_noise.size), prototype_noise)
plt.title("SDM (M=%d)" % m)
plt.xlabel("Exemplars")
plt.ylabel("Percent Corruption")

plt.subplot(1, 3, 2)
plt.imshow(
    1-hop_prototype, 
    cmap='gray', interpolation='nearest', vmin=0.5, vmax=1)
plt.xticks(np.arange(prototype_k.size), prototype_k)
plt.yticks(np.arange(prototype_noise.size), [])
plt.title("Hopfield")
plt.xlabel("Exemplars")

plt.subplot(1, 3, 3)
diff = hop_prototype-sdm_prototype
plt.imshow(
    np.sign(diff)*(1.5**np.log(np.abs(diff))), 
    cmap='RdBu', interpolation='nearest', vmin=-0.5, vmax=0.5)
plt.xticks(np.arange(prototype_k.size), prototype_k)
plt.yticks(np.arange(prototype_noise.size), [])
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


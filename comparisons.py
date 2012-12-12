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

util.save("figures/capacity.svg", width=8, height=6, f_close=False)

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

util.save("figures/tolerance.svg", width=8, height=6, f_close=False)

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

vmin = 0.97
vmax = 1
plt.clf()

gs = plt.GridSpec(1, 3, width_ratios=[10, 10, 1])

plt.subplot(gs[0])
im = plt.imshow(
    1-sdm_prototype, 
    cmap='gray', interpolation='nearest', 
    vmin=vmin, vmax=vmax,
    origin='lower')
plt.xticks(np.arange(prototype_k.size), prototype_k)
plt.yticks(np.arange(noise.size), noise)
plt.title("SDM (M=%d)" % m)
plt.xlabel("Exemplars")
plt.ylabel("Exemplar Percent Corruption")

plt.subplot(gs[1])
plt.imshow(
    1-hop_prototype, 
    cmap='gray', interpolation='nearest', 
    vmin=vmin, vmax=vmax,
    origin='lower')
plt.xticks(np.arange(prototype_k.size), prototype_k)
plt.yticks(np.arange(noise.size), [])
plt.title("Hopfield")
plt.xlabel("Exemplars")

sp = plt.subplot(gs[2])
plt.colorbar(im, cax=sp)
sp.set_ylabel("Prototype Accuracy")

plt.suptitle("Prototype Retrieval from Stored Exemplars", fontsize=15)

util.save("figures/prototype.svg", width=8, height=6, f_close=False)

# <codecell>

diff = hop_prototype-sdm_prototype
diff = np.sign(diff)*(1.5**np.log(np.abs(diff)))
diffmax = np.max(np.abs(diff))
plt.clf()

gs = plt.GridSpec(1, 2, width_ratios=[20, 1])

plt.subplot(gs[0])
im = plt.imshow(
    diff,
    cmap='RdBu', interpolation='nearest', 
    vmin=-diffmax, vmax=diffmax,
    origin='lower')
plt.xticks(np.arange(prototype_k.size), prototype_k)
plt.yticks(np.arange(noise.size), [])
plt.xlabel("Exemplars")

sp = plt.subplot(gs[1])
cb = plt.colorbar(im, sp)
cb.set_ticks([-diffmax, 0, diffmax])
#sp.set_ticks([-diffmax, 0, diffmax])
sp.set_yticklabels(['Hopfield better',
		    'No difference', 
		    'SDM better'])

plt.suptitle("Difference between Hopfield and SDM prototype retrieval", fontsize=15)

util.save("figures/prototype-diff.svg", width=8, height=6, f_close=False)

# <codecell>

import scipy.io as io

b = 20
m = M[-1]
nex = 9

data = io.loadmat('numbers.mat')
zero = data['zero']
one = data['one']
two = data['two']
three = data['three']
four = data['four']
five = data['five']
six = data['six']

inputs = np.hstack([zero,one,two,three,four,five,six])

vec = six.copy()
cvecs = util.corrupt(vec * np.ones((n, nex), dtype='i4'), b)
ex = util.corrupt(vec.copy(), b)

mem1 = sdm.SDM(n, m, D)
mem1.writeM(cvecs, cvecs)
r1 = mem1.readM(ex)

mem2 = hop.hopnet(cvecs)
r2 = mem2.readM(ex, 1000)

for i in xrange(nex):
    plt.clf()
    plt.imshow(cvecs[:, i].reshape((16, 16), order='F'),
	       cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    plt.xticks([], [])
    plt.yticks([], [])
    save("vi_ex%d" % i)

plt.clf()
util.plot_io(ex.reshape((16, 16), order='F'),
	     r1.reshape((16, 16), order='F'))
save("vi_sdm")

plt.clf()
util.plot_io(ex.reshape((16, 16), order='F'),
	     1-r2.reshape((16, 16), order='F'))
save("vi_hop")

# <codecell>

import scipy.io as io
import numpy as np
import matplotlib.pyplot as plt
import sdm as sdm
from util import corrupt, plot_io, save

#data = io.loadmat('sequencePatterns.mat')
#face = data['face']
#x = data['x']
#hi = data['hi']
#a = data['a']
#b = data['b']
#c = data['c']
#d = data['d']
#e = data['e']
#f = data['f']
#g = data['g']
#h = data['h']
#i = data['i']

#inputs = np.hstack([face, x, hi, a, b, c, d, e, f, g, h, i])


data = io.loadmat('numbers.mat')
zero = data['zero']
one = data['one']
two = data['two']
three = data['three']
four = data['four']
five = data['five']
six = data['six']

inputs = np.hstack([zero,one,two,three,four,five,six])



#clean the input to be 0/1
inputs = ((inputs + 1)/2).astype('i4')

# length of input patterns, number of unique patterns
lenPatterns, numPatterns = inputs.shape

# figure size (sqrt(lenPatterns))
figSize = (np.sqrt(lenPatterns)).astype('i4')

# number of sequences ( this is similar to # of exemplars )
numSequences = 10

# percentage of bits to corrupt
numCorrupt = 25


# number of addresses in SDM
numStorage = 10000

# hamming radius
D = 112


reload(sdm)
mem = sdm.SDM(lenPatterns, numStorage, D)
print "Addresses in hamming radius:", mem._select(inputs).sum(axis=0)



for curSeq in xrange(numSequences):
    # store copy of inputs in addresses
    addresses = inputs.copy()

    # corrupt the patterns
    for curPat in xrange(numPatterns):
        a = addresses[:,curPat].copy()
        d = corrupt(addresses[:, curPat], numCorrupt)
        addresses[:, curPat] = d
#        plt.figure(curPat)
#        plot_io(a.reshape((figSize, figSize), order='F'), d.reshape((figSize,figSize), order='F'))


    # write data to address curPat with address curPat+1
    data = np.empty((lenPatterns, numPatterns)).astype('i4')
    data[:, :-1] = addresses[:, 1:]
    data[:, -1] = addresses[:, -1]
        
    mem.writeM(addresses, data)

    # plot what we just wrote
    for i in xrange(numPatterns):
        plt.figure(50+curSeq)
        plt.subplot(1,numPatterns,i+1)
        plt.imshow((1-addresses[:,i]).reshape((figSize,figSize), order='F'),cmap='gray',interpolation='nearest')
        plt.xticks([],[])
        plt.yticks([],[])

    figName="storedSeq"+str(curSeq)
    #save(figName, ext="png", close=True, verbose=True)


# now read sequentially with no noise used as input
#addresses = inputs.copy()
#for curPat in xrange(numPatterns):
#    a = addresses[:,curPat].copy()
#    d = mem.read(addresses[:,curPat]).reshape((figSize, figSize), order='F')
#    plt.figure(numPatterns+curPat)
#    plot_io(addresses[:,curPat].reshape((figSize, figSize), order='F'), d)



#
# read until converge, using retrieved data as the next address
#
# first initialize some things
addresses = inputs.copy()
lastTrial = np.zeros((lenPatterns,1))
# just start with a particular corrupted pattern someone in the sequence
d = corrupt(addresses[:,0],25)
i=0
startAddress = d

# loop till convergence
while (((d == lastTrial).all() == False)):
    # save lastTrial
    lastTrial = d
    i+=1   
 
    # use lastTrial as the address to read from
    d = mem.read(lastTrial)
#    plt.figure(i)
#    plot_io(lastTrial.reshape((figSize, figSize), order='F'),d.reshape((figSize, figSize), order='F'))


# how long this look to converge
numTrialsTillConverged = i

d=startAddress
for i in xrange(numTrialsTillConverged):

    plt.figure(0)
    plt.subplot(1,numTrialsTillConverged,i+1)
    plt.imshow((1-d).reshape((figSize,figSize), order='F'),cmap='gray',interpolation='nearest')
    plt.xticks([],[])
    plt.yticks([],[])

    lastTrial = d
    d=mem.read(lastTrial)


figName="finalSeq"
#save(figName, ext="png", close=True, verbose=True)
    


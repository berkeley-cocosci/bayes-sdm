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
    save(figName, ext="png", close=True, verbose=True)


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
save(figName, ext="png", close=True, verbose=True)
    

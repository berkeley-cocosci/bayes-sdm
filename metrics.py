import numpy as np
import util
import scipy.stats
import sdm
import hopfield as hop

from joblib import Memory
memory = Memory(cachedir="cache", mmap_mode='c', verbose=0)

@memory.cache
def test_hopfield_capacity_n(n, k=1, iters=100):
    corruption = np.empty((iters, k))

    # store the same number of items multiple times
    for i in xrange(iters):
        # generate random inputs
        vecs = util.random_input(n, k)
        v = (vecs * 2) - 1
        # create hopfield net
        mem = hop.hopnet(v)
        # read the items backout
        r = ((mem.readM(v, 1000) + 1) / 2.0).astype('i4')
        # find the largest fraction of corrupted bits
        corruption[i] = np.mean(r ^ vecs, axis=0)

    return corruption

@memory.cache
def test_sdm_capacity_n(params, k=1, iters=100):
    n, m, D = params
    mem = sdm.SDM(n, m, D)
    corruption = np.empty((iters, k))
    
    # store the same number of items multiple times
    for i in xrange(iters):
        # generate random inputs
        vecs = util.random_input(n, k)
        # reset the memory to its original state
        mem.reset()
        # write random inputs to memory
        mem.writeM(vecs, vecs)
        # read the items back out
        r = mem.readM(vecs)
        # find the largest fraction of corrupted bits
        corruption[i] = np.mean(r ^ vecs, axis=0)
        
    return corruption

#@memory.cache(ignore=['verbose'])
def test_capacity(params, k=1, iters=100, thresh=0, verbose=False):
    if hasattr(params, '__iter__'):
        testfunc = test_sdm_capacity_n
    else:
        testfunc = test_hopfield_capacity_n
    
    # test storing different numbers of items
    while True:
        # compute capacity
        corruption = testfunc(
            params, k=k, iters=iters)
        
        # compute statistics about the distances
        maxc = np.max(corruption, axis=1)
        mean = np.mean(maxc)
        sem = scipy.stats.sem(maxc)
        if verbose:
            print "%2d:  %.2f +/- %.2f" % (k, mean, sem)

        k += 1

        # stop once we start to hit corruption
        if mean > thresh:
            break

    return k

@memory.cache
def test_hopfield_noise_tolerance_n(n, k=1, noise=0, iters=100):
    corruption = np.empty((iters, k))
    bits = int(n * noise)

    # store the same number of items multiple times
    for i in xrange(iters):
        # generate random inputs
        vecs = util.random_input(n, k)
        cvecs = util.corrupt_k(vecs, bits)
        v = (vecs * 2) - 1
        cv = (cvecs * 2) - 1
        # create hopfield net
        mem = hop.hopnet(v)
        # read the items backout
        r = ((mem.readM(cv, 1000) + 1) / 2.0).astype('i4')
        # find the largest fraction of corrupted bits
        corruption[i] = np.mean(r ^ vecs, axis=0)

    return corruption

@memory.cache
def test_sdm_noise_tolerance_n(params, k=1, noise=0, iters=100):
    n, m, D = params
    mem = sdm.SDM(n, m, D)
    corruption = np.empty((iters, k))
    bits = int(n * noise)
    
    # store the same number of items multiple times
    for i in xrange(iters):
        # generate random inputs
        vecs = util.random_input(n, k)
        cvecs = util.corrupt_k(vecs, bits)
        # reset the memory to its original state
        mem.reset()
        # write random inputs to memory
        mem.writeM(vecs, vecs)
        # read the items back out
        r = mem.readM(cvecs)
        # find the largest fraction of corrupted bits
        corruption[i] = np.mean(r ^ vecs, axis=0)
        
    return corruption

#@memory.cache(ignore=['verbose'])
def test_noise_tolerance(params, k=1, noise=0, iters=100, thresh=0, verbose=False):
    if hasattr(params, '__iter__'):
        testfunc = test_sdm_noise_tolerance_n
    else:
        testfunc = test_hopfield_noise_tolerance_n
    
    # test storing different numbers of items
    while True:
        # compute noise tolerance
        corruption = testfunc(
            params, k=k, noise=noise, iters=iters)
        
        # compute statistics about the distances
        maxc = np.max(corruption, axis=1)
        mean = np.mean(maxc)
        sem = scipy.stats.sem(maxc)
        if verbose:
            print "%2d:  %.2f +/- %.2f" % (k, mean, sem)

        k += 1

        # stop once we start to hit corruption
        if mean > thresh:
            break

    return k
        


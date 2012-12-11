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
        # create hopfield net
        mem = hop.hopnet(vecs)
        # read the items backout
        r = mem.readM(vecs, 1000)
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

def test_capacity(params, k0=1, iters=100, thresh=0, verbose=False):
    if hasattr(params, '__iter__'):
        testfunc = test_sdm_capacity_n
    else:
        testfunc = test_hopfield_capacity_n

    data = []
    k = k0
        
    # test storing different numbers of items
    while True:
        # compute capacity
        corruption = testfunc(
            params, k=k, iters=iters)
        
        # compute statistics about the distances
        maxc = np.mean(corruption, axis=1)
        mean = np.mean(maxc)
        sem = scipy.stats.sem(maxc)
        data.append((mean, sem))
        if verbose:
            print "%2d:  %.3f +/- %.3f" % (k, mean, sem)

        k += k0

        # stop once we start to hit corruption
        if (mean-sem) > thresh:
            break

    data = np.array(data)

    return data

######################################################################
    
@memory.cache
def test_hopfield_noise_tolerance_n(n, k=1, noise=0, iters=100):
    if noise == 0:
        return test_hopfield_capacity_n(n, k=k, iters=iters)
    
    corruption = np.empty((iters, k))
    bits = int(n * noise)

    # store the same number of items multiple times
    for i in xrange(iters):
        # generate random inputs
        vecs = util.random_input(n, k)
        cvecs = util.corrupt(vecs, bits)
        # create hopfield net
        mem = hop.hopnet(vecs)
        # read the items backout
        r = mem.readM(cvecs, 1000)
        # find the largest fraction of corrupted bits
        corruption[i] = np.mean(r ^ vecs, axis=0)

    return corruption

@memory.cache
def test_sdm_noise_tolerance_n(params, k=1, noise=0, iters=100):
    if noise == 0:
        return test_sdm_capacity_n(params, k=k, iters=iters)
    
    n, m, D = params
    mem = sdm.SDM(n, m, D)
    corruption = np.empty((iters, k))
    bits = int(n * noise)
    
    # store the same number of items multiple times
    for i in xrange(iters):
        # generate random inputs
        vecs = util.random_input(n, k)
        cvecs = util.corrupt(vecs, bits)
        # reset the memory to its original state
        mem.reset()
        # write random inputs to memory
        mem.writeM(vecs, vecs)
        # read the items back out
        r = mem.readM(cvecs)
        # find the largest fraction of corrupted bits
        corruption[i] = np.mean(r ^ vecs, axis=0)
        
    return corruption

def test_noise_tolerance(params, k0=1, noise=0, iters=100, thresh=0, verbose=False):
    if hasattr(params, '__iter__'):
        testfunc = test_sdm_noise_tolerance_n
    else:
        testfunc = test_hopfield_noise_tolerance_n

    data = []
    k = k0
        
    # test storing different numbers of items
    while True:
        # compute noise tolerance
        corruption = testfunc(
            params, k=k, noise=noise, iters=iters)
        
        # compute statistics about the distances
        maxc = np.mean(corruption, axis=1)
        mean = np.mean(maxc)
        sem = scipy.stats.sem(maxc)
        data.append((mean, sem))
        if verbose:
            print "%2d:  %.3f +/- %.3f" % (k, mean, sem)

        k += k0

        # stop once we start to hit corruption
        if (mean-sem) > thresh:
            break

    data = np.array(data)

    return data

######################################################################
        
@memory.cache
def test_hopfield_prototype_n(n, kp=1, ke=1, noise=0, iters=100):
    corruption = np.empty((iters, kp))
    bits = int(n * noise)

    # store the same number of items multiple times
    for i in xrange(iters):
        # generate random inputs
        vecs = util.random_input(n, kp)
        cvecs = vecs[..., None] * np.ones((n, kp, ke), dtype='i4')
        cvecs = util.corrupt(
            cvecs.reshape((n, kp*ke)),
            bits)
        ex = util.corrupt(vecs, bits)
        # create hopfield net
        mem = hop.hopnet(cvecs)
        # read the items backout
        r = mem.readM(ex, 1000)
        # find the largest fraction of corrupted bits
        corruption[i] = np.mean(r ^ vecs, axis=0)

    return corruption

@memory.cache
def test_sdm_prototype_n(params, kp=1, ke=1, noise=0, iters=100):
    n, m, D = params
    mem = sdm.SDM(n, m, D)
    corruption = np.empty((iters, kp))
    bits = int(n * noise)
    
    # store the same number of items multiple times
    for i in xrange(iters):
        # generate random prototype and exemplars
        vecs = util.random_input(n, kp)
        cvecs = vecs[..., None] * np.ones((n, kp, ke), dtype='i4')
        cvecs = util.corrupt(
            cvecs.reshape((n, kp*ke)),
            bits)
        ex = util.corrupt(vecs, bits)
        # reset the memory to its original state
        mem.reset()
        # write random inputs to memory
        mem.writeM(cvecs, cvecs)
        # read the items back out
        r = mem.readM(ex)
        # find the fraction of corrupted bits
        corruption[i] = np.mean(r ^ vecs, axis=0)

    return corruption

def test_prototype(params, kp=1, ke=1, noise=0, iters=100, verbose=False):
    if hasattr(params, '__iter__'):
        testfunc = test_sdm_prototype_n
    else:
        testfunc = test_hopfield_prototype_n
    
    # compute noise tolerance
    corruption = testfunc(
        params, kp=kp, ke=ke, noise=noise, iters=iters)
        
    # compute statistics about the distances
    maxc = np.mean(corruption, axis=1)
    mean = np.mean(maxc)
    sem = scipy.stats.sem(maxc)

    if verbose:
        print "%2d %2d:  %.3f +/- %.3f" % (kp, ke, mean, sem)

    return mean

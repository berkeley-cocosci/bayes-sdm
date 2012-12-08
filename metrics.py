import numpy as np
import util
import scipy.stats
import sdm

from joblib import Memory
memory = Memory(cachedir="cache", mmap_mode='c', verbose=1)

@memory.cache
def test_capacity_n(n, m, D, k=1, iters=100, seed=0):
    rso = np.random.RandomState(seed)
    mem = sdm.SDM(n, m, D, seed=rso)
    corruption = np.empty((iters, k))
    
    # store the same number of items multiple times
    for i in xrange(iters):
        # reset the memory to its original state
        mem.reset()
        # write random inputs to memory
        vecs = util.random_input(n, k, rso=rso)
        mem.writeM(vecs, vecs)
        # read the items back out
        r = mem.readM(vecs)
        # find the largest fraction of corrupted bits
        corruption[i] = np.mean(r ^ vecs, axis=0)
        
    return corruption
        
#@memory.cache(ignore=['verbose'])
def test_capacity(n, m, D, k=1, maxk=0, iters=100, thresh=0, seed=0, verbose=False):

    # test storing different numbers of items
    while k <= maxk:
        # compute capacity
        corruption = test_capacity_n(
            n, m, D, k=k, iters=iters, seed=seed)
        
        # compute statistics about the distances
        mean = np.mean(corruption)
        sem = scipy.stats.sem(corruption)
        if verbose:
            print "%2d:  %.2f +/- %.2f" % (k, mean, sem)

        k += 1

        # stop once we start to hit corruption
        if mean > thresh:
            break

    return k

def test_noise_tolerance(mem, iters, rso, verbose=False):
    n = mem.n
    out = []
    corruption = np.empty(iters)
    
    # test storing different numbers of items
    k = 1
    while True:

	# store the same number of items multiple times
	for i in xrange(iters):
	    # reset the memory to its original state
	    mem.reset()
	    # generate random inputs
	    vecs = util.random_input(n, k, rso=rso)
	    # write the inputs to memory
	    mem.writeM(vecs, vecs)
	    # read the items back out
	    r = mem.readM(vecs)
	    # compute the median fraction of corrupted bits
	    corruption[i] = np.max(np.mean(r ^ vecs, axis=0))

	# compute statistics about the distances
	mean = np.mean(corruption)
	sem = scipy.stats.sem(corruption)
	if verbose:
	    print "%2d:  %.2f +/- %.2f" % (k, mean, sem)

	out.append((mean, sem))
	k += 1

	# stop once we start to hit corruption
	if np.round(mean, decimals=2) > 0:
	    break

    out = np.array(out)
    return out

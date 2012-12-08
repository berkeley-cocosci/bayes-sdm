import numpy as np

def test_capacity(mem, iters, rso, verbose=False):
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

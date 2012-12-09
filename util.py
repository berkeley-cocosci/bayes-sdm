import numpy as np
import matplotlib.pyplot as plt

def corrupt(I, n):
    """Corrupts the pattern 'I' by flipping exactly `n` bits.

    Python version of Bruno Olshausen's `corrupt.m` pythonized by Paul
    Ivanov (and then further modified by Jessica Hamrick).
    
    """
    N = np.size(I)
    idx = np.arange(N)
    np.random.shuffle(idx)
    i = idx[:n][:, None]
    Inew = I.copy()
    Inew[i] *= -1
    return Inew

def plot_io(input, output):
    kwargs = {
        'vmin': 0,
        'vmax': 1,
        'cmap': 'gray',
        'interpolation': 'nearest'
        }
    plt.clf()
    plt.subplot(1, 2, 1)
    plt.imshow(input, **kwargs)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.title("Input")
    plt.subplot(1, 2, 2)
    plt.imshow(output, **kwargs)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.title("Output")

def random_input(n, k=0, rso=None):
    """Generate `k` random binary vectors of length `n`. 

    If k=0, then the output will be a vector of shape (n,). If k>0,
    then the output will be a matrix of shape (n, k).

    Parameters
    ----------
    n   : vector length
    k   : number of vectors (default=0)
    rso : random number object (default=None)

    Returns
    -------
    out : np.ndarray of shape (n,) or (n, k)
      
    """

    # determine the size of the array
    if k == 0:
        size = (n,)
    elif k > 0:
        size = (n, k)
    else:
        raise ValueError("k < 0")

    # generate random vectors
    if rso:
        out = rso.randint(0, 2, size=size)
    else:
        out = np.random.randint(0, 2, size=size)

    return out

def corrupt_k(I, b):
    """Corrupt inputs I by flipping exactly `b` bits for each input.

    Parameters
    ----------
    I : array of shape (n, k)
    b : integer <= n

    """

    n, k = I.shape
    idx = range(n)
    cI = I.copy()
    for i in xrange(k):
        cidx = np.random.choice(idx, size=b, replace=False)
        cI[cidx, i] = 1-cI[cidx, i]
    return cI

def set_fig_properties():
    fig = plt.gcf()
    fig.clf()
    fig.set_figwidth(6)
    fig.set_figheight(4)

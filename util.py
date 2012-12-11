import numpy as np
import matplotlib.pyplot as plt
import random
import os

def corrupt(I, n):
    """Corrupts the pattern 'I' by flipping exactly `n` bits.

    Python version of Bruno Olshausen's `corrupt.m` pythonized by Paul
    Ivanov (and then further modified by Jessica Hamrick).
    
    """
    cI = I.copy()
    if I.ndim == 1:
        cI = I.copy()[:, None]
    N, k = cI.shape
    idx = np.arange(N)
    for i in xrange(k):
        # cidx = np.random.choice(idx, size=n, replace=False)
        cidx = random.sample(idx,n)
        cI[cidx, i] = 1 - cI[cidx, i]
    if I.ndim == 1:
        cI = cI[:, 0]
    return cI

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

def set_fig_properties():
    fig = plt.gcf()
    fig.clf()
    fig.set_figwidth(6)
    fig.set_figheight(4)



def save(path, ext='png', close=True, verbose=True):
    """Save a figure from pyplot.

    Parameters
    ----------
    path : string
        The path (and filename, without the extension) to save the
        figure to.

    ext : string (default='png')
        The file extension. This must be supported by the active
        matplotlib backend (see matplotlib.backends module).  Most
        backends support 'png', 'pdf', 'ps', 'eps', and 'svg'.

    close : boolean (default=True)
        Whether to close the figure after saving.  If you want to save
        the figure multiple times (e.g., to multiple formats), you
        should NOT close it in between saves or you will have to
        re-plot it.

    verbose : boolean (default=True)
        Whether to print information about when and where the image
        has been saved.

    """
    
    # Extract the directory and filename from the given path
    directory = os.path.split(path)[0]
    filename = "%s.%s" % (os.path.split(path)[1], ext)
    if directory == '':
        directory = '.'

    # If the directory does not exist, create it
    if not os.path.exists(directory):
        os.makedirs(directory)

    # The final path to save to
    savepath = os.path.join(directory, filename)

    if verbose:
        print("Saving figure to '%s'..." % savepath),

    # Actually save the figure
    plt.savefig(savepath)
    
    # Close it
    if close:
        plt.close()

    if verbose:
        print("Done")


import numpy as np
import matplotlib.pyplot as plt

def corrupt(I, n):
    """Corrupts the pattern 'I' by flipping exactly `n` bits.

    Python version of Bruno Olshausen's `corrupt.m` pythonized by Paul
    Ivanov (and then further modified by Jessica Hamrick).
    
    """
    N=np.size(I)
    idx = np.arange(N)
    np.random.shuffle(idx)
    i = idx[:n][:, None]
    Inew = I.copy()
    Inew[i]*=-1
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

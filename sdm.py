"""sdm.py -- Sparse Distributed Memory

Authors:
  Jessica Hamrick (jhamrick@berkeley.edu)
  Josh Abbott (josh.abbott@berkeley.edu)

"""

import numpy as np

class SDM(object):

    def __init__(self, n, m, D, seed=0):
        """Initialize a SDM.

        Parameters
        ----------
        n    : length of inputs/addresses
        m    : number of addresses/storage locations
        D    : hamming radius
        seed : random number generator seed

        """

        # save parameters
        self.n = n
        self.m = m
        self.D = D

        # random number generator
        if isinstance(seed, int):
            rso = np.random.RandomState(seed)
        else:
            rso = seed
        
        # address matrix: random binary matrix, where the kth row
        # is the address of the kth storage location
        self.A = rso.randint(0, 2, (m, n, 1))
        self._A = self.A.copy()
        # counter matrix: stores contents of the addressed locations
        self.C = np.zeros((n, m))
        self._C = self.C.copy()

    def reset(self):
        """Reset the SDM to it's original state."""
        self.A = self._A.copy()
        self.C = self._C.copy()
        
    def _select(self, address):
        """Select addresses with the Hamming radius(self.D) of the
        given address

        Parameters
        ----------
        address : vector of size n

        Returns
        -------
        theta : binary vector of size m

        """
        x = np.sum(self.A ^ address, axis=1)
        theta = x <= self.D
        return theta

    def readM(self, addresses):
        """Read the data at the locations indicated by the given M
        addresses.

        Parameters
        ----------
        addresses : array of size (n, M)

        Returns
        -------
        data : array of size (n, M)

        """
        s = self._select(addresses)
        h = np.dot(self.C, s)
        data = np.zeros(h.shape, dtype='i4')
        data[h > 0] = 1
        return data

    def read(self, address):
        """Read the data at the location indicated by the given
        address.

        Parameters
        ----------
        address : vector of size n

        Returns
        -------
        data : vector of size n

        """
        data = self.readM(address[:, None])[:, 0]
        return data

    def writeM(self, addresses, data):
        """Write M vectors of `data` at the locations indicated by the
        given M `addresses`.

        Parameters
        ----------
        addresses : matrix of size (n, M)
        data      : matrix of size (n, M)

        """
        s = self._select(addresses)
        w = (data * 2) - 1
        c = np.sum(w[:, None] * s[None, :], axis=2)
        self.C += c

    def write(self, address, data):
        """Write the given data at the location indicated by the given
        address.

        Parameters
        ----------
        address : vector of size n
        data    : vector of size n

        """
        self.writeM(address[:, None], data[:, None])
